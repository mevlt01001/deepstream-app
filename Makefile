################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2019-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
################################################################################

CUDA_VER?=
ifeq ($(CUDA_VER),)
  $(error "CUDA_VER is not set")
endif

APP:= libdeepstream-app.so
BUILD_DIR:= build

TARGET_DEVICE = $(shell gcc -dumpmachine | cut -f1 -d -)

NVDS_VERSION:=7.1

LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/
APP_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/bin/

ifeq ($(TARGET_DEVICE),aarch64)
  CFLAGS:= -DPLATFORM_TEGRA
endif

SRCS:= $(wildcard *.c) $(wildcard *.cpp)
SRCS+= $(wildcard /opt/nvidia/deepstream/deepstream-7.1/sources/apps/apps-common/src/*.c)
SRCS+= $(wildcard /opt/nvidia/deepstream/deepstream-7.1/sources/apps/apps-common/src/deepstream-yaml/*.cpp)

INCS:= $(wildcard *.h)

PKGS:= gstreamer-1.0 gstreamer-video-1.0 x11 json-glib-1.0

VPATH := $(sort $(dir $(SRCS)))

OBJS := $(addprefix $(BUILD_DIR)/, $(notdir $(patsubst %.c, %.o, $(filter %.c, $(SRCS)))))
OBJS += $(addprefix $(BUILD_DIR)/, $(notdir $(patsubst %.cpp, %.o, $(filter %.cpp, $(SRCS)))))

CFLAGS+= -fPIC -I./ -I/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/sources/apps/apps-common/includes \
         -I/opt/nvidia/deepstream/deepstream-7.1/sources/includes -DDS_VERSION_MINOR=1 -DDS_VERSION_MAJOR=5 \
         -I /usr/local/cuda-$(CUDA_VER)/include

LIBS:= -L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart

LIBS+= -L$(LIB_INSTALL_DIR) -lnvdsgst_meta -lnvds_meta -lnvdsgst_helper -lnvdsgst_customhelper \
      -lnvdsgst_smartrecord -lnvds_utils -lnvds_msgbroker -lm -lyaml-cpp \
    -lcuda -lgstrtspserver-1.0 -ldl -Wl,-rpath,$(LIB_INSTALL_DIR)

CFLAGS+= $(shell pkg-config --cflags $(PKGS))

LIBS+= $(shell pkg-config --libs $(PKGS))

all: $(BUILD_DIR)/$(APP)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: %.c $(INCS) Makefile | $(BUILD_DIR)
	$(CC) -c -o $@ $(CFLAGS) $<

$(BUILD_DIR)/%.o: %.cpp $(INCS) Makefile | $(BUILD_DIR)
	$(CXX) -c -o $@ $(CFLAGS) $<

$(BUILD_DIR)/$(APP): $(OBJS) Makefile
	$(CXX) -shared -o $@ $(OBJS) $(LIBS)

install: $(BUILD_DIR)/$(APP)
	cp -rv $(BUILD_DIR)/$(APP) $(APP_INSTALL_DIR)

clean:
	rm -rf $(BUILD_DIR)