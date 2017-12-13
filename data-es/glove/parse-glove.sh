#!/bin/bash
head -$1 SBW-vectors-300-1MM.txt | awk  '{ for (i = 1; i <= 101; i++)
                                            printf "%s ",$i
                                            print ""
                                            }'

