#!/bin/sh
echo 'will generate mtz files for all hklfiles present'
echo 'please edit cell parameters'
echo 'Syntax: ./generate_mtzs.sh PREFIX_FOR_MTZ_FILE'

PREFIX=$1

for FILE in *.txt; do

    OUTFILE=`echo $FILE | sed -e 's/\.txt$/.mtz/'`
    TMPHKL=`echo $FILE | sed -e 's/\.txt$/.temp.hkl/'`

    echo " Input: $FILE"
    echo "Output: $OUTFILE"
    if [ -e $TMPHKL -o -e $OUTFILE ]; then
        echo "   I'm about to write to the following files, but one or more"
        echo "   of them already exist:"
        echo "   " $OUTFILE
        echo "   " $TMPHKL
        echo "   To confirm that you want to continue, which will DESTROY the"
        echo "   current contents of these files, type 'y' and press enter."
        read conf
        if [ $conf != y ]; then
                echo "Not confirmed."
                exit 1
        else
                echo "Proceeding"
        fi
    fi

    grep -v "End of reflections" $FILE > $TMPHKL

    echo "Running 'f2mtz'..."
    f2mtz HKLIN $TMPHKL HKLOUT ${PREFIX}${OUTFILE} > out.html << EOF
TITLE Reflections from CrystFEL
NAME PROJECT wibble CRYSTAL wibble DATASET wibble
#CELL 62.32   62.32  111.1  90.00  90.00 120.00
CELL 62.37   62.37  111.2  90.00  90.00 120.00
SYMM P63
LABOUT H K L IMEAN SIGIMEAN
CTYPE  H H H J     Q
EOF
done

rm *.temp.hkl
