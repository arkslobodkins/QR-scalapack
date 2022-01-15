OBJS =  main.o get_time.o

all: qr

qr: $(OBJS)
	-mpif90 -O3 -o qr.x $(OBJS) libscalapack.a -llapack -lblas -lm

clean :
	rm -f $(OBJS) qr.x

.f.o : ; mpif90 -c -O3 $*.f

.c.o : ; mpicc -c -DAdd_ -O3 $*.c

