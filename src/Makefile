CPPS = node.cc \
       nn_context.cc \
       layer.cc 
 
default:$(CPPS)
	for CPP in $(CPPS) ; do \
		echo g++ -c $$CPP; \
		g++ -c -I../include/ $$CPP; \
	done
