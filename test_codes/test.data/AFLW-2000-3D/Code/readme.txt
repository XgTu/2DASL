The AFLW2000-3D provides the fitted 3D faces of the first 2000 AFLW samples (https://lrs.icg.tugraz.at/research/aflw/).
It can be used in two ways:

With Basel Face Model (3DMM)
1. Applying for the Basel Face Model (BFM) on "http://faces.cs.unibas.ch/bfm/main.php?nav=1-0&id=basel_face_model".
2. Copy the "01_MorphableModel.mat" file in the BFM to ModelGeneration/.
3. Run the "ModelGenerate.m" to generate the shape model "Model_Shape.mat" and copy it to current folder. Note that this model is the same as our former work in "http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/HPEN/main.htm".
4. Run the "main_show_with_BFM.m".

Without Basel Face Model
Since the application of BFM needs some time, we also provide the 68 3D-landmarks for easy use.
1. Run the 'main_show_without_BFM.m'