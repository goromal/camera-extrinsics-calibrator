#!/bin/bash

CMD="astyle --options=.astylerc"

$CMD src/*
$CMD src/test/*
$CMD include/*
$CMD include/test/*
