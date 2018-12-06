#!/usr/bin/env bash
allennlp train experiment/simple_attention.json -s store/valid --include-package my_library