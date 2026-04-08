# Compilers gcc for Deepstream g++ for ours
CXX = g++
CC = gcc

CXXFLAGS = -std=c++17 -O3 -D_GLIBCXX_USE_CXX11_ABI=1 -fPIC -DPLATFORM_TEGRA -DDS_VERSION_MINOR=1 -DDS_VERSION_MAJOR=5
CFLAGS = -O3 -fPIC -DPLATFORM_TEGRA -DDS_VERSION_MINOR=1 -DDS_VERSION_MAJOR=5

# build directory path (to make workspace more readable)
BUILD_DIR = build

# We used CONDA since we used python torch includes (-I$(CONDA_ENV)/lib/python3.10/site-packages/torch/include/torch/csrc/api/include) 
# and its compiled libraries ($(CONDA_ENV)/lib/python3.10/site-packages/torch.libs)
CONDA_ENV = /home/orin2/miniconda3/envs/libtorch
TORCH_ROOT = $(CONDA_ENV)/lib/python3.10/site-packages/torch
TORCH_LIBS = $(CONDA_ENV)/lib/python3.10/site-packages/torch.libs

# Deepstream Root directory, we used detect and got NvDsMeta struct data from.
DS_ROOT = /opt/nvidia/deepstream/deepstream

# to find gstreamer libraries
PKGS = gstreamer-1.0 gstreamer-video-1.0 x11 json-glib-1.0 glib-2.0

# Include paths
INCLUDES = \
    -I./include \
    -I. \
    -I$(TORCH_ROOT)/include \
    -I$(TORCH_ROOT)/include/torch/csrc/api/include \
    -I$(DS_ROOT)/sources/apps/apps-common/includes \
    -I$(DS_ROOT)/sources/includes \
    -I/usr/local/cuda/include \
    $(shell pkg-config --cflags $(PKGS))

# Library paths
LIB_DIRS = \
    -L$(TORCH_ROOT)/lib \
    -L$(TORCH_LIBS) \
    -L$(CONDA_ENV)/lib \
    -L$(DS_ROOT)/lib \
    -L/usr/local/cuda/lib64

# Runtime paths, helping the system to locate compiled libraries when it is running a compiled file (target) in a terminal.
LDFLAGS = \
    -Wl,-rpath,$(TORCH_ROOT)/lib \
    -Wl,-rpath,$(TORCH_LIBS) \
    -Wl,-rpath,$(CONDA_ENV)/lib \
    -Wl,-rpath,$(DS_ROOT)/lib

# libraries to be linked
LIBS = \
    -ltorch -ltorch_cpu -lc10 \
    -lnvdsgst_meta -lnvds_meta -lnvds_utils \
    -lnvdsgst_helper -lnvdsgst_customhelper -lnvdsgst_smartrecord -lnvds_msgbroker \
    -lgstrtspserver-1.0 -lcuda -lcudart -lyaml-cpp -lm -ldl \
    $(shell pkg-config --libs $(PKGS))

# Source codes
SRCS  = $(wildcard src/*.cpp)
SRCS += $(wildcard src/*.c)
# Deepstream source codes
SRCS += $(wildcard $(DS_ROOT)/sources/apps/apps-common/src/*.c)
SRCS += $(wildcard $(DS_ROOT)/sources/apps/apps-common/src/deepstream-yaml/*.cpp)
# Virtual paths, to makefile understand where the files in
VPATH = src:$(DS_ROOT)/sources/apps/apps-common/src:$(DS_ROOT)/sources/apps/apps-common/src/deepstream-yaml

# compiled object files, compliler generates them in BUILD_DIR
OBJS  = $(addprefix $(BUILD_DIR)/, $(notdir $(patsubst %.cpp, %.o, $(filter %.cpp, $(SRCS)))))
OBJS += $(addprefix $(BUILD_DIR)/, $(notdir $(patsubst %.c, %.o, $(filter %.c, $(SRCS)))))

TARGET = $(BUILD_DIR)/libAI_class.so

all: $(TARGET)

# Lİnker phase
$(TARGET): $(OBJS)
	@mkdir -p $(BUILD_DIR)
	# Burada $(CXX) yani g++ kullanıyoruz ki C++ standart kütüphaneleri sisteme bağlansın.
	$(CXX) -shared $(CXXFLAGS) -o $@ $^ $(LIB_DIRS) $(LDFLAGS) $(LIBS)

# compiling phase for c++
$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# compiling phase for c
$(BUILD_DIR)/%.o: %.c
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# clean (make clean)
clean:
	rm -rf $(BUILD_DIR)

git_update:
	@echo "Be careful!"
	@echo "git reset --hard"
	@echo "git fetch --all"
	@echo "git pull"