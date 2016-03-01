CC = clang++
CCFLAGS = -v -Wall
LDFLAGS = -I/usr/local/include
LBFLAGS = -L/usr/local/lib
LIBS = -lm -lopencv_calib3d -lopencv_core -lopencv_features2d -lopencv_flann -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_ml -lopencv_objdetect -lopencv_photo -lopencv_shape -lopencv_stitching -lopencv_superres -lopencv_ts -lopencv_video -lopencv_videoio -lopencv_videostab
TARGET = objectTracking

all: $(TARGET)

$(TARGET): $(TARGET).o
	$(CC) $(LIBS) $(CCFLAGS) -o $@ $^

$(TARGET).o: $(TARGET).cpp
	$(CC) $(CCFLAGS) -c $<

.PHONY: clean

clean:
	rm -rf *.o core* $(TARGET)
