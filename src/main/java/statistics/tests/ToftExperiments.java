package statistics.tests;
import fileIO.*;

public class ToftExperiments {
	
	public static void main(String[] args) {
		double[][] data;
		double[][]	tSig;
		double[][]	rrsSig;
		double[][]	mwSig;
		String path="C:/Research/Data/Toft Data/";
		InFile f = new InFile(path+"Test 1.csv");
		int nosSamples=6, nosPerSample=100;
		data= new double[nosSamples][nosPerSample];
		for(int j=0;j<nosPerSample;j++)
			for(int i=0;i<nosSamples;i++)
				data[i][j]=f.readDouble();
		tSig=new double[nosSamples][nosSamples];
		rrsSig=new double[nosSamples][nosSamples];
		mwSig=new double[nosSamples][nosSamples];
		for(int i=0;i<nosSamples;i++)
		{
			for(int j=0;j<nosSamples;j++)
			{
				if(i!=j)
				{
					tSig[i][j]=TwoSampleTests.studentT_TestStat(data[i],data[j]);
					rrsSig[i][j]=TwoSampleTests.rrs_TestStat(data[i],data[j]);
					mwSig[i][j]=TwoSampleTests.mw_TestStat(data[i],data[j]);
				}
			}
		}
		OutFile of = new OutFile(path+"Results 1.csv");
		of.writeString("T Test Sig Figures \n");
		for(int i=0;i<nosSamples;i++)
		{	//nosSamples
			for(int j=0;j<nosSamples;j++)
				of.writeString(tSig[i][j]+",");
			of.writeString("\n");
		}
		of.writeString("MW Sig Figures \n");
		for(int i=0;i<nosSamples;i++)
		{
			for(int j=0;j<nosSamples;j++)
				of.writeString(mwSig[i][j]+",");
			of.writeString("\n");
		}
		of.writeString("RRS Sig Figures \n");
		for(int i=0;i<nosSamples;i++)
		{
			for(int j=0;j<nosSamples;j++)
				of.writeString(rrsSig[i][j]+",");
			of.writeString("\n");
		}
	}
}
