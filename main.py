#!/usr/bin/env python3

import sys

from audioCheck import check

def main():
    check(*sys.argv[1:])

main()