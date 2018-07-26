all:
	gcc -g trj_analysis.c -std=c99 -o out -O3 -DQUIET

# can also use:
# -fassociative-math
# -ftree-vectorize (automatically included becuse of -ffast-math/ -Ofast)
# -ftree-vectorizer-verbose=[n] gives report on auto-vectorisation (which loops were good)
# see CompilerAutovectorizationGuide.pdf
opt:
	gcc trj_analysis11.c -o out -Ofast -std=c99 -g -DQUIET -pipe -march=corei7 -mtune=corei7 -mfpmath=sse -msse -msse2 -msse3 -msse4.2 -funroll-loops -fforce-addr

valgrind:
	valgrind --dsymutil=yes --leak-check=full --show-leak-kinds=all --track-origins=yes ./out

callgrind:
	# need to use clang :(
	cc -g trj_analysis.c -std=c99 -o out -O3 -DQUIET
	valgrind --dsymutil=yes --tool=callgrind --instr-atstart=yes ./out
	mv callgrind.out.* callgrind.out
	callgrind_annotate callgrind.out trj_analysis.c | grep -v "???"

clean:
	rm -rf *.dSYM
	rm -rf callgrind.out*
