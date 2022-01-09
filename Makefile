OBJS = pdgels.o main.o get_time.o

all: pdgels

pdgels: $(OBJS)
	-mpif90 -O3 -o pdgels.out $(OBJS) libscalapack.a -llapack -lblas -lm

clean :
	rm -f $(OBJS) pdgels.out

.f.o : ; mpif90 -c -O3 $*.f

.c.o : ; mpicc -c -DAdd_ -O3 $*.c

