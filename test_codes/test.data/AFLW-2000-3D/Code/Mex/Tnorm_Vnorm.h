

void TNorm2VNorm(double* normv, int nver, double* normt, double* tri, int ntri)
{
	int i,j;
	for(i = 0; i < nver; i++)
	{
		normv[3*i] = normv[3*i+1] = normv[3*i+2] = 0;
	}

	for(i = 0; i < ntri; i++)
	{
		int pt1 = tri[3*i] - 1;
		int pt2 = tri[3*i+1] - 1;
		int pt3 = tri[3*i+2] - 1;

		for(j = 0; j < 3; j++)
		{
			normv[3*pt1+j] += normt[3*i+j];
			normv[3*pt2+j] += normt[3*i+j];
			normv[3*pt3+j] += normt[3*i+j];
		}

	}
}