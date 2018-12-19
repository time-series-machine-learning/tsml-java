package transformations;
import fileIO.*;
import statistics.tests.ResidualTests;

public class VarianceStabalisingStepwiseRegression {
	static int m,n;
	static 	double[][] X;
	static 	double[][] transformedX;
	static 	double[][] workingX;
	static 	double[] Y;
	static 	double[] transformedY;
	static double[]   powers;
	static	boolean[] included;
	static	boolean[] transformIncluded;
	static int[] positions;
	static int size=0;
	static int[] transformedPositions;
	static int transSize=0;
	static final double CRITICAL=7;
	//0.823;
	
	public static void main(String[] args) {
//Each one should load X and Y scaling if necessary		
		OutFile f = new OutFile("C:/Research/Data/Gavin Competition/Results/StepwiseTransformationPowerResults.csv");
		f.writeLine("Synthetic,MSE,AP,KS,RT,RQ");
		int choice=2;
		Synthetic();
		if(choice==0)
			fullModel(f);
		else if (choice==1)
			stepwiseLinear(f);
		else
			forwardSelectTransform(f);
/*		System.out.println(" Starting Temp Full ...");
		f.writeLine("Temp Full");
		Temperature();
		if(choice==0)
			fullModel(f);
		else if (choice==1)
			stepwiseLinear(f);
		else
			forwardSelectTransform(f);
*/		System.out.println(" Starting Temp Reduced ...");
		f.writeLine("Temp Reduced");
		TemperatureReduced();
		if(choice==0)
			fullModel(f);
		else if (choice==1)
			stepwiseLinear(f);
		else
			forwardSelectTransform(f);
				System.out.println(" Starting SO2 ...");
		f.writeLine("SO2");
		SO2();
		if(choice==0)
			fullModel(f);
		else if (choice==1)
			stepwiseLinear(f);
		else
			forwardSelectTransform(f);
		System.out.println(" Starting SO2 Reduced...");
		f.writeLine("SO2 Reduced");
		SO2Reduced();
		if(choice==0)
			fullModel(f);
		else if (choice==1)
			stepwiseLinear(f);
		else
			forwardSelectTransform(f);
/*		System.out.println(" Starting Precip...");
		f.writeLine("Precip");
		Precip();
		if(choice==0)
			fullModel(f);
		else if (choice==1)
			stepwiseLinear(f);
		else
			forwardSelectTransform(f);
		System.out.println(" Starting Precip Reduced...");
*/		f.writeLine("Precip Reduced");
		PrecipReduced();
		if(choice==0)
			fullModel(f);
		else if (choice==1)
			stepwiseLinear(f);
		else
			forwardSelectTransform(f);
		
	}
	public static void Synthetic() {
		m=1;
		n=256;
		int n2=128;
		int synthScale=5;
		String path="C:/Research/Data/Gavin Competition/Synthetic/";
		String p1="Synthetic Train.csv";
		String p2="Synthetic Validate.csv";
		boolean finished=false;
		int attCount=0, c;
		double var,minVar=Double.MAX_VALUE,newVar, oldSSR, newSSR;
		X=new double[m+1][n];
		Y=new double[n];
		transformedY=new double[n];
		InFile f = new InFile(path+p1);
		for(int i=0;i<n;i++)
			X[0][i]=1;
		for(int i=0;i<n;i++)
		{
			for(int j=1;j<=m;j++)
				X[j][i]=f.readDouble()+synthScale;
			Y[i]=f.readDouble();
		}
//1: Fit linear model, estimate S^2, 
		
	}

	public static void TemperatureReduced() {
		m=20;
		n=7117;
		double tempScale=10;
		String path="C:/Research/Data/Gavin Competition/Temperature/TempTransformed Train.csv";
//		String path="C:/Research/Data/Gavin Competition/Temperature/TempTraining.csv";
//Attributes to remove
		int[] collinear= {1,3,4,5,6,7,20,34,35,36,47,48,72,82};
		boolean finished=false;
		int attCount=0, c;
		double var,minVar=Double.MAX_VALUE,newVar, oldSSR, newSSR;
		X=new double[m+1][n];
		Y=new double[n];
		transformedY=new double[n];
		
		InFile f = new InFile(path);
		for(int i=0;i<n;i++)
			X[0][i]=1;
		for(int i=0;i<n;i++)
		{
			for(int j=1;j<=m;j++)
				X[j][i]=f.readDouble()+tempScale;
			Y[i]=f.readDouble();
		}
		int c1=0,c2=0;

	
	}
	public static void Temperature() {
		m=106;
		n=7117;
		double tempScale=10;
		String path="C:/Research/Data/Gavin Competition/Temperature/TempTraining.csv";
//Attributes to remove
		int[] collinear= {1,3,4,5,6,7,20,34,35,36,47,48,72,82};
		boolean finished=false;
		int attCount=0, c;
		double var,minVar=Double.MAX_VALUE,newVar, oldSSR, newSSR;
		X=new double[m+1][n];
		Y=new double[n];
		transformedY=new double[n];
		
		InFile f = new InFile(path);
		for(int i=0;i<n;i++)
			X[0][i]=1;
		for(int i=0;i<n;i++)
		{
			for(int j=1;j<=m;j++)
				X[j][i]=f.readDouble()+tempScale;
			Y[i]=f.readDouble();
		}
		int c1=0,c2=0;

		double[][] reducedX=new double[m+1-collinear.length][];
		for(int i=0;i<=m;i++)
		{
			if(c1>=collinear.length || i!=collinear[c1])
			{
				reducedX[c2]=X[i];
				c2++;
			}
			else
				c1++;
		}
		X=reducedX;
		m=reducedX.length-1;
	
	}
	
	public static void SO2() {
		m=26;
		n=15304;
		int so2Scale=10;
		String path="C:/Research/Data/Gavin Competition/SO2/";
		String p1="SO2Train.csv";
		boolean finished=false;
		int attCount=0, c;
		double var,minVar=Double.MAX_VALUE,newVar, oldSSR, newSSR;
		X=new double[m+1][n];
		Y=new double[n];
		transformedY=new double[n];
		InFile f = new InFile(path+p1);
		for(int i=0;i<n;i++)
			X[0][i]=1;
		for(int i=0;i<n;i++)
		{
			for(int j=1;j<=m;j++)
				X[j][i]=f.readDouble()+so2Scale;
			Y[i]=f.readDouble();
		}
	}
	public static void SO2Reduced() {
		m=19;
		n=15304;
		int so2Scale=10;
		String path="C:/Research/Data/Gavin Competition/SO2/";
		String p1="SO2TrainReduced.csv";
		boolean finished=false;
		int attCount=0, c;
		double var,minVar=Double.MAX_VALUE,newVar, oldSSR, newSSR;
		X=new double[m+1][n];
		Y=new double[n];
		transformedY=new double[n];
		InFile f = new InFile(path+p1);
		for(int i=0;i<n;i++)
			X[0][i]=1;
		for(int i=0;i<n;i++)
		{
			for(int j=1;j<=m;j++)
				X[j][i]=f.readDouble()+so2Scale;
			Y[i]=f.readDouble();
		}
	}

	public static void Precip() {
		m=106;
		n=7031;
		int precipScale=6;
		String path="C:/Research/Data/Gavin Competition/Precipitation/";
		String p1="PrecipitationTrain.csv";
		boolean finished=false;
		int attCount=0, c;
		double var,minVar=Double.MAX_VALUE,newVar, oldSSR, newSSR;
		X=new double[m+1][n];
		Y=new double[n];
		transformedY=new double[n];
		InFile f = new InFile(path+p1);
		for(int i=0;i<n;i++)
			X[0][i]=1;
		for(int i=0;i<n;i++)
		{
			for(int j=1;j<=m;j++)
				X[j][i]=f.readDouble()+precipScale;
			Y[i]=f.readDouble();
		}
	}
	public static void PrecipReduced() {
		m=20;
		n=7031;
		int precipScale=6;
		String path="C:/Research/Data/Gavin Competition/Precipitation/";
		String p1="PrecipTrainReduced.csv";
		boolean finished=false;
		int attCount=0, c;
		double var,minVar=Double.MAX_VALUE,newVar, oldSSR, newSSR;
		X=new double[m+1][n];
		Y=new double[n];
		transformedY=new double[n];
		InFile f = new InFile(path+p1);
		for(int i=0;i<n;i++)
			X[0][i]=1;
		for(int i=0;i<n;i++)
		{
			for(int j=1;j<=m;j++)
				X[j][i]=f.readDouble()+precipScale;
			Y[i]=f.readDouble();
		}
	}
	
	
	static public void findStats(OutFile f2, LinearModel lm)
	{
		double s = lm.findStats();
		double[] resids=lm.getResiduals();
		double[] pred=lm.getPredicted();
		double ap=ResidualTests.anscombeProcedure(pred,resids);
		double ks=ResidualTests.kolmogorovSmirnoff(resids);
		double rt=ResidualTests.runsTest(pred,resids);
		double gq=ResidualTests.goldfeldQuandt(X,Y,1);
		
		System.out.println("YJ, s^2 = "+s+", AP = "+ap+", KS = "+ks+", RT = "+rt+", GQ = "+gq);
		f2.writeLine("FullReg,"+s+","+ap+","+ks+","+rt+","+gq);
	
	}
	public static void fullModel(OutFile f2)
	{
		LinearModel lm = new LinearModel(X,Y);
		lm.fitModel();
		findStats(f2,lm);
		//2. Fit YJ
		double best = YeoJohnson.findBestTransform(X,Y);
		System.out.println(" Best Transform = "+best);		
		double[] newY = YeoJohnson.transform(best,Y);
		lm = new LinearModel(X,newY);
		lm.fitModel();
		f2.writeLine("YJ Transform");
		findStats(f2,lm);
	}
	
	

	public static void stepwiseLinear(OutFile f2)
	{
		boolean finished=false;
		included=new boolean[m+1];
		int attCount=0;
		double var,oldSSR,newSSR,newVar;
	//This is going to record whether the base values are included, not the transformed		
		included[0]=true;	//Always include constant
		for(int i=1;i<=m;i++)
			included[i]=false;
		positions=new int[m+1];
		while(!finished)
		{
			attCount=formatRegressors();
			//Fit linear model with current candidates
			LinearModel lm = new LinearModel(workingX,Y);
			lm.fitModel();
			var=lm.findStats();
			oldSSR=lm.getSSR();
//			System.out.println(" Att Count = "+attCount+" SSE plain Y = "+var);
	//Try adding in each variable in position attCount, record best improvement
	//Fitting one more model the necessary, but makes code clearer
	//At the end, workingX should have the new candidate in position attCount			
	//Returns the POSITION IN X			
//			int bestPos=findBestAdditionTransformed(attCount);
			int bestPos=findBestAddition(attCount);
			attCount++;
	//If improvement significant, add in permanently by setting flag, otherwise dont
			lm = new LinearModel(workingX,Y);
			lm.fitModel();
			newVar=lm.findStats();
			newSSR=lm.getSSR();
			System.out.println(" Verification: New Var = "+newVar);
//			System.out.println("SSR Change = "+(newSSR-oldSSR));
//			System.out.println("Test stat = "+(newSSR-oldSSR)/var);
			
			if((newSSR-oldSSR)/var>CRITICAL)
			{
				System.out.println("ADDING = "+bestPos);
				included[bestPos]=true;
				positions[attCount-1]=bestPos;
				size=attCount;
			}
			else
			{
				System.out.println("NOT ADDING = "+bestPos);
				included[bestPos]=false;
				finished=true;
				
			}
			if(attCount==m+1)
				finished=true;
//  Try removing any already in the model, if no significant worsening, remove			
			int worst;
			if(attCount>3)
			{
				worst=tryRemovals(X[bestPos],newVar, newSSR);
				if(worst!=-1)
				{
					System.out.println(" Removing Element > "+worst);
					included[worst]=false;
					int x=0;
					while(x<attCount)
					{
						if(positions[x]==worst)
						{
							while(x<attCount-1)
							{
								positions[x]=positions[x+1];
								x++;
							}
						}
						x++;	
					}
					attCount--;
					attCount=formatRegressors();			
				}
			}
		}	
		//Get full daignositics on final model
		attCount=formatRegressors();
		//Fit linear model with current candidates
		LinearModel lm = new LinearModel(workingX,Y);
		lm.fitModel();
		findStats(f2,lm);
			
	}
	
	static int count=0;
	public static void forwardSelectTransform(OutFile f2)
	{
		boolean finished=false, useYJ=false;
		double bestLambda=1,temp;
		included=new boolean[m+1];
		int attCount=0;
		LinearModel lm;
		double var,oldSSR,newSSR,newVar;
	//This is going to record whether the base values are included, not the transformed		
		included[0]=true;	//Always include constant
		for(int i=1;i<=m;i++)
			included[i]=false;
		positions=new int[m+1];
		powers=new double[m+1];
		
		while(!finished)
		{
			attCount=formatRegressors();
			//Fit linear model with current candidates
			lm = new LinearModel(workingX,Y);
			lm.fitModel();
			var=lm.findStats();
			oldSSR=lm.getSSR();
//			System.out.println(" Att Count = "+attCount+" SSE plain Y = "+var);
	//Try adding in each variable in position attCount, record best improvement
	//Fitting one more model the necessary, but makes code clearer
	//At the end, workingX should have the new candidate in position attCount			
	//Returns the POSITION IN X			
			int bestPos=findBestAdditionTransformed(attCount);
//			int bestPos=findBestAddition(attCount);
			attCount++;
	//If improvement significant, add in permanently by setting flag, otherwise dont
			lm = new LinearModel(workingX,Y);
			lm.fitModel();
			newVar=lm.findStats();
			newSSR=lm.getSSR();
			System.out.println(" Verification: New Var = "+newVar);
//			System.out.println("SSR Change = "+(newSSR-oldSSR));
//			System.out.println("Test stat = "+(newSSR-oldSSR)/var);
			
			if((newSSR-oldSSR)/var>CRITICAL)
			{
				System.out.println("ADDING = "+bestPos);
				included[bestPos]=true;
				positions[attCount-1]=bestPos;
				size=attCount;
			}
			else
			{
				System.out.println("NOT ADDING = "+bestPos);
				included[bestPos]=false;
				finished=true;
				
			}
			if(attCount==m+1)
				finished=true;
			attCount=formatRegressors();
			//Yeo Johnson first
			System.out.println(" TRY YJ: ");
			bestLambda=YeoJohnson.findBestTransform(workingX,Y);
//		Round to the nearest 0.5
			temp=((double)Math.round(bestLambda*2))/2;
			double alpha=1;
			System.out.println("Best Lambda value ="+bestLambda+" Rounded = "+temp);
			int p=0;
			useYJ=false;
			if(temp!=1)
			{
				transformedY=YeoJohnson.transform(temp,Y);
				lm=new LinearModel(workingX,transformedY);
				lm.fitModel();
				double s=lm.findInverseStats(temp,Y);
				useYJ=true;
				System.out.println("s = "+s);
			}

		
		}	
		//Get full daignositics on final model
		attCount=formatRegressors();
		//Fit linear model with current candidates
		if(useYJ)
		{
			temp=((double)Math.round(bestLambda*2))/2;
			transformedY=YeoJohnson.transform(temp,Y);
			lm = new LinearModel(workingX,transformedY);
		}
		else
			lm = new LinearModel(workingX,Y);
		lm.fitModel();
		findStats(f2,lm);
		OutFile f3 = new OutFile("TestTrans"+count+".csv");
		count++;
		for(int i=0;i<powers.length;i++)
			f3.writeString(powers[i]+",");
		for(int j=0;j<X[0].length;j++)
		{	
			for(int i=0;i<X.length;i++)
				f3.writeString(X[i][j]+",");
			f3.writeString("\n");
		}
	}
	
	
	public static void transformCode()
{
	/*		//Simple parameter search on variable just entered
	attCount=formatRegressors();
	int p=0;
	double alpha;
	while(p<size && positions[p]!=bestPos) p++;
	System.out.println("p = "+p+" Pos = "+positions[p]);
	if(p<size)
	{
			alpha=PowerSearch.transformRegressor(workingX,Y,p);
			System.out.println("Alpha = "+alpha);
			alpha =((double) Math.round(alpha*2))/2.0;
			System.out.println("Rounded Alpha = "+alpha);
			
			if(alpha==0)
			{
				for(int i=0;i<X[bestPos].length;i++)
					X[bestPos][i]=Math.log(X[bestPos][i]);
			}
			else if (alpha!=1)
			{
				for(int i=0;i<X[bestPos].length;i++)
					X[bestPos][i]=Math.pow(X[bestPos][i],alpha);
			}
	}
//
//		First effort, try YJ and B-T on newly entered
	
//	Try Transformations, not going to bother Y with TEMP
		attCount=formatRegressors();
	//Yeo Johnson first
	System.out.println(" Nos atts = "+attCount);
	double bestLambda=YeoJohnson.findBestTransform(workingX,Y);
//Round to the nearest 0.5
	double temp=((double)Math.round(bestLambda*2))/2;
	double alpha=1;
	System.out.println("Best Lambda value ="+bestLambda+" Rounded = "+temp);
	int p=0;
	boolean yo=false;
	if(temp!=1)
	{
		transformedY=YeoJohnson.transform(temp,Y);
		lm=new LinearModel(workingX,transformedY);
		lm.fitModel();
		double s=lm.findInverseStats(temp,Y);
		yo=true;
		System.out.println("s = "+s);
	}
	while(p<size && positions[p]!=bestPos) p++;
	System.out.println("p = "+p+" Pos = "+positions[p]);
	if(p<size)
	{
		if(yo)
			alpha=BoxTidwell.transformRegressor(workingX,transformedY,p);
		else
			alpha=BoxTidwell.transformRegressor(workingX,Y,p);
	}
	System.out.println("*************  ALPHA = "+alpha);
*/

}
	
	
//Calculate SSR of removing each one except a, then only remove if not significantly worse
	public static int tryRemovals(double[] inData, double var, double fullSSR)
	{
		int worst=0,outPos;
		double worstSSR=Double.MAX_VALUE,ssr,s;
//Swap new one into position 0, always include
		LinearModel lm;
		double[][] temp = new double[size-1][];
		temp[0]=workingX[0];
		temp[1]=inData;
		double[] out=workingX[1];
		double[] t;
		int[] tempPos=new int[size-1];
		int tempOutPos,a;
		tempPos[0]=0;
		tempOutPos=positions[1];
		tempPos[1]=positions[size-1];
//		for(int i=0;i<size;i++)
//			System.out.println(" Position = "+positions[i]);
		System.out.println(" size = "+size);
		for(int i=2;i<size-1;i++)
		{
			temp[i]=workingX[i];
			tempPos[i]=positions[i];
		}

//		System.out.println(" Removing element "+tempOutPos);
//		for(int i=0;i<size-1;i++)
//			System.out.println(" Temp position = "+tempPos[i]);
		int i=2;
		do{
			//Fit reduced model
			lm=new LinearModel(temp,Y);
			lm.fitModel();
			//Find new SSR, record the largest of reduced
			s=lm.findStats();
			ssr=lm.getSSR();
//			System.out.println(" SSR when removing "+(i-1)+ " which has original position "+positions[(i-1)]+" is = "+ssr+" with s^2="+s);
			if(ssr<worstSSR)
			{
				worstSSR=ssr;
				worst=i-1;
			}
			//Swap attribute in and out if done
			if(i<size-1)
			{
				a=tempOutPos;
				tempOutPos=tempPos[i];
//				System.out.println(" Removing element "+tempOutPos+" Adding element "+a+" back in");
				tempPos[i]=a;
//				for(int j=0;j<size-1;j++)
//					System.out.println(" Temp position = "+tempPos[j]);
				t=temp[i];
				temp[i]=out;
				out=t;
			}
			i++;
		}while(i<size);
//Test worst, if not significant, return position. 
//NOTE that the position in the ORIGINAL data recorded by positions[worst]
		outPos=positions[worst];

		if(((fullSSR-worstSSR)/var)<CRITICAL)
			return outPos;
		return -1;
	}	
	public static int findBestAdditionTransformed(int a)
	{
		int best=-1;
		double[][] temp = new double[a+1][];
		LinearModel lmTemp;
		double minSSE=Double.MAX_VALUE,s,bestPower=1;
        System.arraycopy(workingX, 0, temp, 0, a);
		for(int i=0;i<included.length;i++)
		{
			if(!included[i])
			{
				temp[a]=X[i];//	new double[X[i].length];
//				for(int j=0;j<X[i].length;j++)
//					temp[a][j]=X[i][j];
				double power=PowerSearch.transformRegressor(temp,Y,a);
				if(power!=1)
					temp[a]=PowerSearch.transform(X[i],power);
				
				lmTemp=new LinearModel(temp,Y);
				lmTemp.fitModel();
				s=lmTemp.findStats();
//				System.out.println(" Adding in attribute = "+i+" with power = "+power+" New MSE = "+s);
				if(s<minSSE)
				{
					minSSE=s;
					best=i;
					bestPower=power;
				}
			}
		}
		if(best>0)	//Shouldnt be 0
		{
			System.out.println(" BEST to add = "+best+" with power = "+bestPower+" MSE = "+minSSE);
			powers[best]=bestPower;
			temp[a]=PowerSearch.transform(X[best],bestPower);
			X[best]=temp[a];
			workingX=temp;
		}
		return best;
	}

		
	public static int findBestAddition(int a)
	{
		int best=-1;
		double[][] temp = new double[a+1][];
		LinearModel lmTemp;
		double minSSE=Double.MAX_VALUE,s;
        System.arraycopy(workingX, 0, temp, 0, a);
		for(int i=0;i<included.length;i++)
		{
			if(!included[i])
			{
				temp[a]=X[i];
				lmTemp=new LinearModel(temp,Y);
				lmTemp.fitModel();
				s=lmTemp.findStats();
//				System.out.println(" Adding in attribute = "+i+" New MSE = "+s);
				if(s<minSSE)
				{
					minSSE=s;
					best=i;
				}
			}
		}
		if(best>0)	//Shouldnt be 0
		{
			System.out.println(" BEST to add = "+best+" with MSE = "+minSSE);
			temp[a]=X[best];
			workingX=temp;
		}
		return best;
	}

	public static int formatRegressors()
	{
		int attCount=0;
		for(int i=0;i<included.length;i++)
			if(included[i]) attCount++;
		workingX= new double[attCount][];
		int c=0;
		for(int i=0;i<included.length;i++)
		{
			if(included[i])
			{
				workingX[c]=X[i];
				positions[c]=i;
				c++;
			}
		}
		return attCount;
	}
	
	
}
