# you can use "$sh run.sh" to run the command in this file easily
# can also type the command in order, and the commands will execute after the previous one is done
# Notice to change the "--screen_num" in order to avoid the data confrontation!!
# 2/29/2020


## THIS IS THE VERSION THAT TEST THE VALIDATED PICTURE AS INPUT!!! (NOT THE DISPLACED VERSION AS BEFORE!!)

read -p "Please set the number of  nFeat: " nFeat
read -p "Please set the number of nResBlock: " nResBlock
read -p "Please set which screen you are currently used: " num_screen
read -p "Please set the number of nEpochs: " num_nEpochs
echo -e " \n INFO: nFeat: ${nFeat}   nResBlock: ${nResBlock}  screen: ${num_screen} nEpochs:  ${num_nEpochs}"

#Train:
python train.py --nFeat ${nFeat} --nResBlock ${nResBlock} --nEpochs ${num_nEpochs} --nTrain 1000 --nVal 3 --cuda --threads 0 --screen_num ${num_screen}

#Test(old)

#python test.py --model model_pretrained/net_F16B2_epoch_30.pth --input_image image_test/LR_onepiece_test_0001.png --output_filename result/F16B2_onepiece_test_0001.png --compare_image ref/HR_onepiece_test_0001.png --cuda --screen_num 0

#test_combo: (useful)

#python test_script_v1.py --model model_pretrained/${num_screen}/net_F${nFeat}B${nResBlock}_epoch_${num_nEpochs}.pth --cuda --combo True --screen_num ${num_screen}

