//
// Copyright 2019 Zacharier
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0//
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <stdarg.h>

#include <filesystem>
#include <fstream>
#include <iostream>

#include "cola/base/io_util.h"
#include "cola/base/logging.h"
#include "cola/base/tensor.h"
#include "cola/base/types.h"
#include "cola/predictor.h"
#include "cola/proto/cola.pb.h"
#include "cola/trainer.h"

struct Options {
  std::string phase;
  std::string config;
  std::string model;
  std::string input;
};

static void Usage(const char* name, const char* msg, ...) {
  if (msg) {
    va_list ap;
    va_start(ap, msg);
    fprintf(stderr, " %s ", name);
    vfprintf(stderr, msg, ap);
    fprintf(stderr, "%s", "\n");

    va_end(ap);
  }
  const char* fmt =
      "Usage: %s [-p PHASE] [-c CONFIG] [-m MODEL] [-i INPUT]\n"
      "Options:\n"
      "   -p       phase: 'infer' or 'train'\n"
      "   -c       config file path\n"
      "   -m       model file path\n"
      "   -i       input file path at 'infer' phase\n"
      "   -h       show this help\n";

  fprintf(stderr, fmt, name);
  exit(0);
}

static Options DoArgs(int argc, const char* argv[]) {
  if (argc == 0) {
    Usage(argv[0], nullptr);
  }
  if (argc < 2) {
    Usage(argv[0], "at least one argument!");
  }
  auto parse = [&](int idx) {
    if (idx == argc) {
      Usage(argv[0], "required value of '%s'", argv[idx - 1]);
    }
    return argv[idx];
  };

  Options options;
  for (int i = 1; i < argc; ++i) {
    if (std::string("-p") == argv[i]) {
      options.phase = parse(++i);
    } else if (std::string("-c") == argv[i]) {
      options.config = parse(++i);
    } else if (std::string("-i") == argv[i]) {
      options.input = parse(++i);
    } else if (std::string("-m") == argv[i]) {
      options.model = parse(++i);
    } else if (std::string("-h") == argv[i]) {
      Usage(argv[0], nullptr);
    } else {
      Usage(argv[0], "unrecognized option: %s", argv[i]);
    }
  }
  if (options.phase == "train") {
    if (options.config.empty()) {
      Usage(argv[0], "required config path");
    }
  } else if (options.phase == "infer") {
    if (options.model.empty()) {
      Usage(argv[0], "required model path");
    }
    if (options.input.empty()) {
      Usage(argv[0], "required input path");
    }
  } else {
    Usage(argv[0], "unrecognized arg: %s", options.phase.c_str());
  }
  return options;
}

int main(int argc, const char* argv[]) {
  Options options = DoArgs(argc, argv);
  if (options.phase == "train") {
    cola::Config config;
    if (!cola::ReadProtoTxt(options.config, &config)) {
      return 1;
    }
    cola::Trainer trainer;
    if (!trainer.Load(config)) {
      return 1;
    }
    trainer.Train();
  } else {
    cola::Predictor predictor;
    if (!predictor.Load(options.model)) {
      return 1;
    }
    std::string content;
    if (!cola::ReadFile(options.input, &content)) {
      return 1;
    }
    auto input = cola::Tensor<cola::Float>::Create(
        (cola::Float*)content.data(),
        {1, content.size() / sizeof(cola::Float)});
    auto output = cola::Tensor<cola::Float>::Create(input.shape());

    predictor.Predict(input, &output);
    for (size_t i = 0; i < output.size(); ++i) {
      std::cout << output.data()[i] << std::endl;
    }
  }
  return 0;
}