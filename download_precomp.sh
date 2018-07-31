#!/bin/bash


echo $(date) "Start to download precomputed data."
echo "Please note that files are huge."
echo "Please choose experiments of interests:"
read -p "10-fold cross-validation (275 GB after compression) [y]/n ? " kfold
read -p "Leave-one-mode-out (9.4 GB after compression) [y]/n ? " lomo
read -p "Time-stamped (22 GB after compression) [y]/n ? " ts
read -p "Prospective (21.6 GB after compression) [y]/n ? " pro

kfold=${kfold:-y}
lomo=${lomo:-y}
ts=${ts:-y}
pro=${pro:-y}

function check_download_file()
{
    FILENAME=$1
    MD5=$2
    CHECK_MD5SUM=$3
    if [ -f $FILENAME ]
    then
        if [[ `$CHECK_MD5SUM $FILENAME` = *"$MD5"* ]]
        then
            echo $(date) "$FILENAME was downloaded."
        else
            echo $(date) "Downloading data $FILENAME"
            curl -O http://static.lichtargelab.org/Diffusion2018/$FILENAME
        fi
    else
        echo $(date) "Downloading data $FILENAME"
        curl -O http://static.lichtargelab.org/Diffusion2018/$FILENAME
    fi

    if [[ `$CHECK_MD5SUM $FILENAME` = *"$MD5"* ]]
    then
        echo $(date) "Data $FILENAME was downloaded completely."
        tar zxvf $FILENAME -C ./data/interim/algorithms/ --strip 1
    else
        echo $(date) "Data $FILENAME was downloaded incompletely."
        echo $(date) "Redownloading $FILENAME is recommended."
        return
    fi
}

CHECK_MD5SUM=`command -v md5sum`
if [[ $CHECK_MD5SUM != *"md5sum"* ]]
then
    CHECK_MD5SUM=`command -v md5`
    if [[ $CHECK_MD5SUM != *"md5"* ]]
    then
        echo "md5 or md5sum is required to check file."
        exit 1
    fi
fi

# check_download_file test.tar.gz ad32be3ed83fa25ec7e22513d4fa13c2 $CHECK_MD5SUM
# exit 1

# $CHECK_MD5SUM data.tar.gz
# check_download_file data.tar.gz 06f5abcb44fc156d795d728d886b04e5 $CHECK_MD5SUM
if [[ `echo $kfold | tr '[:upper:]' '[:lower:]'` = "y"* ]]
then
    check_download_file 10_fold.tar.gz a9b538536c0b59f3a3860ac77fffa755 $CHECK_MD5SUM
fi

if [[ `echo $ts | tr '[:upper:]' '[:lower:]'` = "y"* ]]
then
    check_download_file time_stamped.tar.gz 16a18fce049820f5cce95d7f5eba3916 $CHECK_MD5SUM
fi

if [[ `echo $pro | tr '[:upper:]' '[:lower:]'` = "y"* ]]
then
    check_download_file prospective.tar.gz 5cbd2158c9b8895cb4aa836e1800ae48 $CHECK_MD5SUM
fi

if [[ `echo $lomo | tr '[:upper:]' '[:lower:]'` = "y"* ]]
then
    check_download_file lomo.tar.gz f996d985a4fa9c6baf02ff91f2021cf2 $CHECK_MD5SUM
fi

# # MD5_DATA=`md5 data.tar.gz`
# if [ -f 'data.tar.gz' ]
# then
#     if [[ `md5 data.tar.gz` = *"bacb6c2b141ad433b7e9a346c950c8a1"* ]]
#     then
#         echo $(date) "data.tar.gz was downloaded."
#     else
#         echo $(date) "Downloading data"
#         curl -O http://static.lichtargelab.org/Diffusion2018/time_stamped.tar.gz
#     fi
# else
#     echo $(date) "Downloading data"
#     curl -O http://static.lichtargelab.org/Diffusion2018/time_stamped.tar.gz
# fi
# tar zxvf data.tar.gz


# NEWLINE=$'\n'

# echo $(date) "Installation was completed.${NEWLINE}Check ./notebooks/ for tutorials."