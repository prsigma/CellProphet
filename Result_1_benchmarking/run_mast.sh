dataset=('/home/pengrui/STGRN/BEELINE-data/inputs/scRNA-Seq/hESC' '/home/pengrui/STGRN/BEELINE-data/inputs/scRNA-Seq/hHep' '/home/pengrui/STGRN/BEELINE-data/inputs/scRNA-Seq/mDC' '/home/pengrui/STGRN/BEELINE-data/inputs/scRNA-Seq/mESC' '/home/pengrui/STGRN/BEELINE-data/inputs/scRNA-Seq/mHSC-E' '/home/pengrui/STGRN/BEELINE-data/inputs/scRNA-Seq/mHSC-GM' '/home/pengrui/STGRN/BEELINE-data/inputs/scRNA-Seq/mHSC-L')

for i in "${dataset[@]}"; do
    /home/pengrui/mambaforge/envs/grn/bin/Rscript /home/pengrui/STGRN/MAST_script.R ${i}
done