"""progress.py - ProgressBar class"""

import time
import sys

class ProgressBar(object):

    instances = []
    
    @property
    def quiet(self):
        return self._quiet

    @quiet.setter
    def quiet(self, state):
        self._quiet = state
        self._active = not state
    
    @property
    def active(self):
        return self._active

    @active.setter
    def active(self, state):
        self._quiet = state
        self._active = not state

    def __init__(self, final, length=50, label='', tail_label='', quiet=False):

        self.quiet = quiet
        self.final = final
        self.length = length
        self.label = label
        self.tail = tail_label
        self.update = self._update # .update() is the exposed interface to users
        self.bar = ''
        self.i = 0
        self.val = ''
        self._t0 = time.time()
        self.instances.append(self)

        self.fmt = '\r[{bar:<%ds}]'%(length+1)
        if len(label):
            self.fmt = '\r[{label:^%ds}][{bar:<%ds}]'%(len(label), length+1)
        if len(tail_label):
            self.fmt += '[{tail:^%ds}]'%(len(tail_label))

        self.rate = 1
        self.unit = 1        
        if final < length:
            self.unit = length//final
        elif length < final:
            self.rate = final//length + (final%length > 0) # ceil

    def _fmt_label(self, label):
        p = self.i*100./self.final                         # {p:} for percentage
        f = '{:d}/{:d}'.format(self.i, self.final)         # {f:} for fraction
        t = int(time.time()-self._t0)                      # {t:} for time
        v = self.val                                       # {v:} for custom val
        label = label.format(p=p, f=f, t=t, i=self.i, v=v) # {i:} for index
        return time.strftime(label)                   # e.g. %d/%m/%y %H:%M:%S

    def _update(self, val=''):
        self.i += 1
        if self.i%self.rate == 0:
            if self.final-self.rate <= self.i < self.final: # penultimate update
                self.update = self._update_final
            self.bar += '=' * self.unit
            self.val = val
            self._write(self.bar + '>')

    def _update_final(self, val=''):
        self.i += 1
        if self.i == self.final:
            bar = '=' * self.length
            self.val = val
            self._write(bar + '>')
            sys.stdout.write('\n')

    def _write(self, bar):
        if self.active:
            label = self._fmt_label(self.label)
            tail = self._fmt_label(self.tail)
            bar = self.fmt.format(label=label, bar=bar, tail=tail)
            sys.stdout.write(bar)
            sys.stdout.flush()