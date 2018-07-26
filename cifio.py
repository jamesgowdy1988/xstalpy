import re
import pickle
import struct

class ParseCIF:

    _num_re = re.compile(r'\A(-?\d+\.?\d*)$')
    _quote_re = re.compile(r'"|\'')
    _split_re = re.compile(r'"[^"]+"|\'[^\']+\'|\S+')
    _oneliner_re = re.compile(r'\n[^\n]?\b_[^\s]+[ \t]+(?!_)[^\n]+')
    _multiliner_re = re.compile(r'\n[^\n]?\b_[^\s\n]+[\s\n]+;.+?\n;', re.DOTALL) # non-greedy
    _loop_re = re.compile(r' ?loop_(?:\n[\s^\n]*\b_[^\n]+)+(?:\n(?!_| _|loop_| loop_)[^\n]*)+')
    _token_re = re.compile(r'\b_[^\n\s]+') # in case of multi token lines
    _data_re = re.compile(r'\n(?!\#|_|\ _)[^\n]+')

    def __init__(self, ciffile):
        ParseCif.parse_cif(self, ciffile)

    def parse_cif(self, ciffile, mode='r'):

        with open(ciffile, mode=mode) as f:
            cif = self.instream = f.read().replace('\r\n','\n')

        # one time only regex compiled above in case multiple class instances
        loops = self._loop_re.findall(cif)
        oneliners = self._oneliner_re.findall(cif)
        multiliners = self._multiliner_re.findall(cif)

        self.tokens = []
        self.data = []
        for loop in loops:
            tokens = map(str.lstrip, self._token_re.findall(loop))
            data = map(self._clean_lines, self._data_re.findall(loop))
            data = map(list, zip(*data)) # transpose
            self.tokens.append(tokens)
            self.data.append(data)
            assert len(tokens) == len(data)

        for line in oneliners:
            token, datum = line.split(None, 1)
            self.tokens.append([token.strip()])
            self.data.append([datum])

        for lines in multiliners:
            token, datum = lines.split(';', 1)
            token = token.strip()
            self.tokens.append([token])
            self.data.append([datum])

        assert len(self.tokens) == len(self.data)

        self.tokensdict = {} 
        for tokens, data in zip(self.tokens, self.data):
            for tok, dat in zip(tokens, data):
                self.tokensdict[tok] = dat[0] if len(dat) == 1 else dat

    def _clean_str(self, v):
        s = self._num_re.search(v)
        choices = {'.': None, 'yes': 1, 'no': 0}
        if s: 
            return float(s.group(1))
        elif v in choices: 
            return choices[v]
        else: 
            return self._quote_re.sub('', v)

    def _clean_lines(self, line):
        line = line.lstrip()
        line = self._split_re.findall(line) # quoted text is not split
        line = [self._clean_str(v) for v in line]
        return line


class ParseCBF(ParseCIF):

    '''Parse a Pilatus imgCBF file compressed with the byte-offset algorithm

    :arg str cbffile: path to a imgCBF file
    :ivar dict tokendict: CIF style tokens and data structure
    :ivar dict cbf_header: the CBF binary data section header
    :ivar dict pilatus_header: the original PILATUS detector header
    :ivar np.ndarray data: intensity counts of pixels stored in a 2D array
    '''

    def __init__(self, cbffile):
        ParseCbf.parse_cbf(self, cbffile)

    def parse_cbf(self, cbffile):

        # first parse the ascii CIF parts
        self.cbffile = cbffile 
        self.parse_cif(cbffile, mode='rb')

        # get (pilatus) header convention
        convention = self.tokensdict['_array_data.header_convention']
        pattern = r'(SLS|PILATUS)_\d+(\.?\d*)*'
        match = re.search(pattern, convention)
        assert match is not None
        convention = match.group(0)

        # parse pilatus header
        if convention == 'PILATUS_1.2':
            header = self.tokensdict['_array_data.header_contents']
            self.parse_pilatus_header(header)
        else:
            msg = 'Don\'t know header convention: {}'.format(convention)
            raise NotImplemented(msg)

        # parse CBF binary section header (assuming has one ie. is CBFimg)
        # also currently self._multiliner_re only parses the first image (;)
        # threfore if multiple CBFs in one file this will break

        cbf = self.tokensdict['_array_data.data']
        self.get_offsets_endian(cbf)
        header = cbf[:self.start]
        self.parse_cbf_header(header)

        # get CBF binary section header values and check them
        padd = self.cbf_header['X-Binary-Size-Padding']
        dim1 = self.cbf_header['X-Binary-Size-Fastest-Dimension']
        dim2 = self.cbf_header['X-Binary-Size-Second-Dimension']
        nelem = self.cbf_header['X-Binary-Number-of-Elements']
        assert nelem == dim1*dim2
        assert 'x-CBF_BYTE_OFFSET' in self.cbf_header['Content-Type']
        
        # parse CBF binary data using byte_offset algorithm
        endtag = '--CIF-BINARY-FORMAT-SECTION----'
        nonbin = cbf.find(endtag)
        start, stop = int(self.start+4), int(self.stop+padd)
        binarydata = cbf[start:stop]
        self.parse_data(binarydata, maxi=nelem)
        assert self._curr == self.cbf_header['X-Binary-Size']
        self.data = self._lineardata.reshape(dim2, dim1)

    def _clean_lines2(self, line):
        line = line.strip(';\t ')
        for oldspace in '#:=,()': 
            line = line.replace(oldspace, ' ')
        return line

    def parse_pilatus_header(self, header):

        with open(get_abspath("pilfmt.pkl"), 'rb') as pilfmt_specfile: 
            # PILFMT from PILATUS format spec by Marcus Mueller at Dectris
            # MTZFMT spec here http://www.ccp4.ac.uk/html/mtzformat.html
            PILFMT, MTZFMT = pickle.load(pilfmt_specfile)

        header = header.split('\n#')   
        self.pilatus_header = {}
        for key, (pattern, idxs, dtype) in PILFMT.iteritems():
            for line in header:
                line = self._clean_lines2(line)
                if re.search(r'\b'+pattern+r'\b', line):
                    line = line.split()
                    vals = []
                    for idx in idxs:
                        if isinstance(idx, slice): 
                            vals.extend(line[idx])
                        else: 
                            vals.append(line[idx])

                    # datatype handling
                    if vals[0] == 'not':
                        vals = float('NaN') if dtype is float else None 
                    elif len(vals) == 1:
                        vals = dtype(vals[0])
                    elif dtype is str:
                        vals = ' '.join(vals)
                    else:
                        vals = map(dtype, vals)

                    self.pilatus_header[key] = vals

    # next two method find points where the binary data starts and stops
    # and decides if we need to swap the byte order (should be little endian)

    def get_offsets_endian(self, cbf, nchars=10e5):
        starttags = {'\x0c\x1a\x04\xd5': '<', '\xd5\x04\x1a\x0c': '>'}

        self.start = nchars
        for tag in starttags:
            i = cbf.find(tag) 
            if 0 < i < self.start:
                self.start = i
                self.end = starttags[tag]

        msg = 'Start tag not in first {} characters?'.format(nchars)
        assert self.start != nchars, msg
        assert struct.unpack(self.end+'i', tag)[0] == -721151476

        self.stop = self.start
        for tag in ['\n', '\r\n']:
            i = cbf.find(tag+'--CIF-BINARY-FORMAT-SECTION----')
            if self.start < i:
                self.stop = i
                break

        assert self.stop != self.start, 'Stop tag not found'

    def _get_offsets_endian_DEBUG(self, cbf):
        from string import printable
        for i, char in enumerate(cbf):
            if char == chr(12): # ord(char) == 12
                self.end = '<'
                break
            elif char == chr(213): # ord(char) == 213
                self.end = '>'
                break
            else:
                msg = 'Non-ascii in header: {}'.format(repr(char))
                assert char in printable, msg

        assert i != len(self.buff)
        tag = cbf[i:i+4]
        assert struct.unpack(self.end+'i', tag)[0] == -721151476
        self.start = i

        for i, char in enumerate(reversed(cbf)):
            if char not in string.printable: 
                self.stop = len(cbf)-i 
                break

        assert hasattr(self, 'stop'), 'Stop tag not found'

    def parse_cbf_header(self, header):
        header = re.split(r'(?<!;)\n', header)
        header = filter(bool, header)
        assert header[0] == '--CIF-BINARY-FORMAT-SECTION--'
        header = dict(map(lambda h: h.split(': '), header[1:]))
        for k, v in header.iteritems(): 
            header[k] = self._clean_str(v)

        self.cbf_header = header

    _nbit_fmt = {8: 'b', 16: 'h', 32: 'i', 64: 'l'} # signed ints
    _nbit_hexmin = {8: '-0x80', 16: '-0x8000', 32: '-0x80000000'}

    def _unpack_DEBUG(fmt, buff):
        # alternative implementation of struct.unpack
        from binascii import hexlify
        if fmt[0] in '><':
            if fmt[0] == '<': 
                # little endian reads hex backwards
                buff = buff[::-1]
            fmt = fmt[1:]
        nbytes = self._nbit_fmt[fmt]/8
        states = (8*2)**(2*nbytes)
        maxval = states/2
        val = int(hexlify(buff), 16)
        if val < maxval: 
            return val
        elif val < 2*maxval: 
            return val-2*maxval # i think

    def _parse_int(self, start, nbits):

        fmt = self.end+self._nbit_fmt[nbits]
        stop = start+nbits//8
        buff = self._buff[start:stop] # if KeyError binary data truncated?
        val, = struct.unpack(fmt, buff)

        # recursively call until correct int type
        if hex(val) == self._nbit_hexmin[nbits]:
            return self._parse_int(stop, nbits*2)
        else:
            return stop, val

    def parse_data(self, buff, maxi=100):
        # byte offset algorithm
        i = curr = pix = 0
        lineardata = []
        self._buff = buff
        while i < maxi:
            i += 1
            curr, val = self._parse_int(curr, 8)
            pix += val
            lineardata.append(pix)

        self._curr = curr
        self._lineardata = np.array(lineardata)
