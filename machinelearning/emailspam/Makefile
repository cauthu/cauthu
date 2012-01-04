# $Id: Makefile 223 2011-12-13 00:37:08Z nguyen59 $

# do some static linkings because we can't compile on hatswitch (can't
# intall libboost-thread), so have to compile on a different machine,
# which has a different netfilterqueue version
#LINKFLAGS = -L/usr/local/lib -Wl,-rpath,/usr/local/lib -lboost_filesystem -lboost_regex

CXXFLAGS = -g -Wall -O0
INCFLAGS = 
LINKFLAGS = -lboost_filesystem -lboost_regex -lhtmlcxx

# for vmime
LINKFLAGS += -lvmime

# for memetic 0.9.7: needs -DHAVE_INTTYPES_H to prevent mimetic's
# error regarding thread_id
# CXXFLAGS += -DUSE_MIMETIC -DHAVE_INTTYPES_H
# LINKFLAGS += -L/tmp/mimetic/lib -lmimetic
# INCFLAGS += -I/tmp/mimetic/include


PREC := $(wildcard *.cc)
TARGETS := $(PREC:%.cc=%)

all : $(TARGETS)

$(TARGETS) : % : %.o
	g++ $(CXXFLAGS) ${INCFLAGS} -o $@ $< $(LINKFLAGS)

%.o : %.cc
	g++ $(CXXFLAGS) ${INCFLAGS} -c $< $(INCFLAGS)

clean :
	rm -f core* *.o $(TARGETS)

