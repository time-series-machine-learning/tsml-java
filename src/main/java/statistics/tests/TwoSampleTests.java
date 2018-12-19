package statistics.tests;
/*Written in the dark and distant past
Algorithms implemented from descriptions in Neave and Worthington

Assumes unpaired samples. For paired samples just take the difference and use
OneSam
*USAGE:
        TwoSampleTests test=new TwoSampleTests();
        double[] d1=...
        double[] d2=...
        String str=test.performTests(d1,d2);
OUTPUT: A comma separated string with testName,testStatistic,pValue (assuming alpha 0.05)

**/
import statistics.distributions.*;
import java.util.*;

public class TwoSampleTests extends Tests{

	public String performTests(double[] a, double[] b)
//Performs t-test (unequal var), Mann-Whitney and Robust Rank Order test

//Returns for each test
//	TestName, Test Statistic, large sample 1-sided p-value, 	
	{
            DataPoint[][] d = new DataPoint[2][];
            d[0]=new DataPoint[a.length];
            d[1]=new DataPoint[b.length];
            for(int i=0;i<a.length;i++)
                    d[0][i]=new DataPoint(a[i],0,i);
            for(int i=0;i<b.length;i++)
                    d[1][i]=new DataPoint(b[i],1,i);
//T test 
            TestResults t_test=new TestResults("T_Test");
            studentT_Test(t_test,d,0);
            t_test.findPValue();
            String str="T_Test,"+t_test.testStat+","+t_test.pValue+"\n";
//Mann-Whitney
            TestResults mw=new TestResults("Mann_Whitney");
            mannWhitney(mw,d);
            mw.findPValue();
            str+="Mann_Whitney,"+mw.testStat+","+mw.pValue+"\n";
//Robust Rank Sum
            TestResults rrs=new TestResults("robustRankSum");
            robustRankSum(rrs,d);
            rrs.findPValue();
            str+="robustRankSum,"+rrs.testStat+","+rrs.pValue+"\n";

            return str;
	}
	public static TestResults performTwoSampleTest(double[] a, double[] b, int testType)
	{
            DataPoint[][] d = new DataPoint[2][];
            d[0]=new DataPoint[a.length];
            d[1]=new DataPoint[b.length];
            for(int i=0;i<a.length;i++)
                    d[0][i]=new DataPoint(a[i],0,i);
            for(int i=0;i<b.length;i++)
                    d[1][i]=new DataPoint(b[i],1,i);
//T test 
            TestResults testR=new TestResults("T_Test");
            switch(testType)
            {
                    case 0:
                            studentT_Test(testR,d,0);
                            break;
                    case 1:
                            mannWhitney(testR,d);
                            break;
                    case 2:	
                            robustRankSum(testR,d);
                            break;
                    default:
                            System.out.println(" Test Not implemented: exit");
                            System.exit(0);
            }
            return testR;
	}
	public static double performTest(double[] a, double[] b, int testType, boolean returnPVal)
	{
            TestResults r=performTwoSampleTest(a,b,testType);
            if(returnPVal)
            {
                r.findPValue();
                return r.pValue;
            }
            else
                return r.testStat;
	}
	
	
	public static double studentT_PValue(double[] a, double[] b)
	{
            DataPoint[][] d = new DataPoint[2][];
            d[0]=new DataPoint[a.length];
            d[1]=new DataPoint[b.length];
            for(int i=0;i<a.length;i++)
                    d[0][i]=new DataPoint(a[i],0,i);
            for(int i=0;i<b.length;i++)
                    d[1][i]=new DataPoint(b[i],1,i);
//T test 
            TestResults t_test=new TestResults("T_Test");
            studentT_Test(t_test,d,0);
            t_test.findPValue();
            return t_test.pValue;
	}
	public static double mw_PValue(double[] a, double[] b)
	{
            DataPoint[][] d = new DataPoint[2][];
            d[0]=new DataPoint[a.length];
            d[1]=new DataPoint[b.length];
            for(int i=0;i<a.length;i++)
                    d[0][i]=new DataPoint(a[i],0,i);
            for(int i=0;i<b.length;i++)
                    d[1][i]=new DataPoint(b[i],1,i);
            TestResults mw=new TestResults("Mann_Whitney");
            mannWhitney(mw,d);
            mw.findPValue();
            return mw.pValue;
	}
	public static double rrs_PValue(double[] a, double[] b)
	{
            DataPoint[][] d = new DataPoint[2][];
            d[0]=new DataPoint[a.length];
            d[1]=new DataPoint[b.length];
            for(int i=0;i<a.length;i++)
                    d[0][i]=new DataPoint(a[i],0,i);
            for(int i=0;i<b.length;i++)
                    d[1][i]=new DataPoint(b[i],1,i);
            TestResults rrs=new TestResults("robustRankSum");
            robustRankSum(rrs,d);
            rrs.findPValue();
            return rrs.pValue;
	}
	
	public static double studentT_TestStat(double[] a, double[] b)
	{
            DataPoint[][] d = new DataPoint[2][];
            d[0]=new DataPoint[a.length];
            d[1]=new DataPoint[b.length];
            for(int i=0;i<a.length;i++)
                    d[0][i]=new DataPoint(a[i],0,i);
            for(int i=0;i<b.length;i++)
                    d[1][i]=new DataPoint(b[i],1,i);
//T test 
            TestResults t_test=new TestResults("T_Test");
            studentT_Test(t_test,d,0);
            return t_test.testStat;
	}
	public static double mw_TestStat(double[] a, double[] b)
	{
            DataPoint[][] d = new DataPoint[2][];
            d[0]=new DataPoint[a.length];
            d[1]=new DataPoint[b.length];
            for(int i=0;i<a.length;i++)
                    d[0][i]=new DataPoint(a[i],0,i);
            for(int i=0;i<b.length;i++)
                    d[1][i]=new DataPoint(b[i],1,i);
            TestResults mw=new TestResults("Mann_Whitney");
            mannWhitney(mw,d);
            return mw.testStat;
	}
	public static double rrs_TestStat(double[] a, double[] b)
	{
            DataPoint[][] d = new DataPoint[2][];
            d[0]=new DataPoint[a.length];
            d[1]=new DataPoint[b.length];
            for(int i=0;i<a.length;i++)
                    d[0][i]=new DataPoint(a[i],0,i);
            for(int i=0;i<b.length;i++)
                    d[1][i]=new DataPoint(b[i],1,i);
            TestResults rrs=new TestResults("robustRankSum");
            robustRankSum(rrs,d);
            return rrs.testStat;
	}

	
	public static void robustRankSum(TestResults t, DataPoint[][] d)
//	Find U as with Mann-Whitley, different test stat
	{
//	1. Find placement arrays for both ways
//		1.1 Merge two data series into one, 
            DataPoint[] mergedD=new DataPoint[d[0].length+d[1].length];
            for(int i=0;i<d[0].length;i++)
                    mergedD[i]=d[0][i];
            for(int i=0;i<d[1].length;i++)
                    mergedD[d[0].length+i]=d[1][i];
            //1.2. Sort combined data series
            Arrays.sort(mergedD);
            int[] p_YX=new int[d[0].length];
            int[] p_XY=new int[d[1].length];
            int m=p_YX.length;
            int n=p_XY.length;
            int j=0;
            int countA=0;
            int countB=0;
            double u_YX=0,u_XY=0;
            for(int i=0;i<mergedD.length; i++)
            {
                    //If d[i] from sample A increment U
                if(mergedD[i].sampleNumber()==0)
                {
                    p_YX[countA]=countB;
                    u_YX+=p_YX[countA];
                    countA++;
                }
                else
                {
                    p_XY[countB]=countA;
                    u_XY+=p_XY[countB];
                    countB++;
                }
            }
            System.out.println(" Series A positions = ");
            for(int i=0;i<p_XY.length;i++)
                    System.out.print(p_XY[i]+",");
            System.out.println(" Series B positions = ");
            for(int i=0;i<p_YX.length;i++)
                    System.out.print(p_YX[i]+",");
            System.out.println(" u1 = "+u_XY+"  u2 = "+u_YX);
            System.out.println(" m*n = "+p_XY.length*p_YX.length+"  u1+u2 = "+(u_XY+u_YX));

//			
//  2. Calculate test statistic
            System.out.println(" U_A statistic = "+u_YX/m);
            System.out.println(" U_B statistic = "+u_XY/n);
            u_YX/=m;
            u_XY/=n;
            double Va=0,Vb=0;
            for(int i=0;i<m;i++)
                    Va+=(p_YX[i]-u_YX)*(p_YX[i]-u_YX);
            for(int i=0;i<n;i++)
                    Va+=(p_XY[i]-u_XY)*(p_XY[i]-u_XY);

            double U=(m*u_YX-n*u_XY)/(2*Math.sqrt(Va+Vb+u_YX*u_XY));
            t.dist = new NormalDistribution(0,1);
            t.testStat=U;
            System.out.println(" U New = "+t.testStat);
		
	}

	public static void wilcoxonRankSum(TestResults t, DataPoint[][] d)
//Find U as with Mann-Whitley, different test stat
	{
		double U=(double)findU(d);
		t.testStat=U+0.5*d[0].length*(d[0].length+1);		
		System.out.println(" R statistic = "+t.testStat);
		
	}
	
/** Man Whitney two sample test
 * H. B. Mann and D. R. Whitney "On a test of whether one of two random variables
 * is stochastically larger than another" Ann. Math. Statist. 18, 50-60 (1947) 
 * 
 * The test on two samples sums the number of elements in sample B that are larger
 * than each element of sample A. This test statistic, U, has what kind of distribution?
 * tables are given in the book.
 * 
 * 
 *  the number of elements 
 * @param t Test results
 * @param d Data set
 * 
 * 
 * Notes: Works with unequal sample sizes
 * Related to Wilcoxon test
 */	
	public static void mannWhitney(TestResults t, DataPoint[][] d)
	{
//1. Find U.		
		t.testStat=(double)findU(d);
//		System.out.println(" U statistic = "+t.testStat);
		double nA=(double)d[0].length;
		double nB=(double)d[1].length;
		double nullMean=0.5*nA*nB;
		double nullStDev=Math.sqrt(nA*nB*(nA+nB+1)/12.0);
		t.dist=new NormalDistribution(nullMean,nullStDev);
		System.out.println(" Null Mean ="+nullMean+" Null SD = "+nullStDev);
//		a*NB = "+d[0].length*d[1].length);

	}
	public static void studentT_Test(TestResults t, DataPoint[][] d, int type){
            switch(type){//Should probably enum this
                case 0: //Default to two sample, unequal variance
                    unequalVarianceStudentT_Test(t,d);
                    break;
                case 1: //two sample, equal variance, NOT IMPLEMENTED
                    unequalVarianceStudentT_Test(t,d);
                    break;
                case 2: //pair two sample   
                    
            }
        }
//Defaults to two sample test assuming unequal variance        
	public static void unequalVarianceStudentT_Test(TestResults t, DataPoint[][] d)
	{
//Find means and var
		double m1=0,m2=0;
		double s1=0,s2=0;
		for(int i=0;i<d[0].length;i++)
			m1+=d[0][i].d;
		for(int i=0;i<d[1].length;i++)
			m2+=d[1][i].d;
		m1/=d[0].length;
		m2/=d[1].length;
		for(int i=0;i<d[0].length;i++)
			s1+=(d[0][i].d-m1)*(d[0][i].d-m1);
		for(int i=0;i<d[1].length;i++)
			s2+=(d[1][i].d-m2)*(d[1][i].d-m2);
		s1/=(d[0].length-1);
		s2/=(d[1].length-1);
		
//Find test stat
		double tStat=(m1-m2)/Math.sqrt((s1/d[0].length+s2/d[1].length));
		t.testStat=tStat;		

//Find df
		int n1=d[0].length;
		int n2=d[1].length;
		
		t.df1=(int)Math.ceil( (s1/n1+s2/n2)*(s1/n1+s2/n2)/( (s1/n1)*(s1/n1)/(n1-1)+ (s2/n2)*(s2/n2)/(n2-1) ) ) ;
		t.dist=new StudentDistribution(t.df1);
	
	}
	
	
	private static DataPoint[] mergeData(DataPoint[][] d)
	{
		DataPoint[] md=new DataPoint[d[0].length+d[1].length];
		for(int i=0;i<d[0].length;i++)
			md[i]=d[0][i];
		for(int i=0;i<d[1].length;i++)
			md[d[0].length+i]=d[1][i];
		return md;	
	}

	
	
	static private int findU(DataPoint[][] d, int a, int b)
	{
//		The test on two samples sums the number of elements in sample B that are 
//		smaller than each element of sample A. 
//NUMBER OF B's proceeding each A
//		1. Merge two data series into one, 
		DataPoint[] mergedD=new DataPoint[d[a].length+d[b].length];
		for(int i=0;i<d[a].length;i++)
			mergedD[i]=d[a][i];
		for(int i=0;i<d[b].length;i++)
			mergedD[d[a].length+i]=d[b][i];
		
//2. Sort combined data series
		Arrays.sort(mergedD);
//3. Find U statistic: Does NOT handle equal values
		int j=0;
		int count=0;
		int U=0;
		for(int i=0;i<mergedD.length && j<d[a].length; i++)
		{
			//If d[i] from sample A increment U
			if(mergedD[i].sampleNumber()==a)
			{
				U+=count;
				j++;
			}
			//else from sample B, increment count
			else
				count++;
		}
		return U;
	}
	static private int findU(DataPoint[][] d)
//Implementation 1: Does not deal with ties: Defaults for finding positions
//	The test on two samples sums the number of elements in sample B that are larger
//	 * than each element of sample A. 
	{
//1. Linear scan to find the number of sample B proceeding sample A

		if(d.length!=2)
		{
			System.out.println("Error, cannot use this for k!=2");
			System.exit(0);
		}
		return findU(d,0,1);
	}
	public static void testTwoSamples()
	/* 
		 * Test Data 1: page 110
		 * Sample A: 	3,7,15,10,4,6,4,7
		 * Sample B:	19,11,36,8,25,23,38,14,17,41,25,21
		n_a = 8, n_b=12
		Test Stat: U=4

		 * Test Data 2: Feltovich paper "Nonparametric tests of differences in medians
		 * Sample A: 	5.025,6.7,6.725,6.75,7.05,7.25,8.375
		 * Sample B:	4.875,5.125,5.225,5.55,5.75,5.925,6.125
		n_a = m=7, n_b=n=8
		Test Stat: U=4
		 *
		 *		 */ 
		{
			int n_a=8;
			int n_b=12;
			DataPoint[][] d = new DataPoint[2][];
			d[0]=new DataPoint[n_a];
			d[1]=new DataPoint[n_b];
			double []d1={3,7,15,10,4,6,4,7};
			for(int i=0;i<n_a;i++)
				d[0][i]=new DataPoint(d1[i],0,i);
			double []d2={19,11,36,8,25,23,38,14,17,41,25,21};
			TwoSampleTests ts = new TwoSampleTests();
			String str = ts.performTests(d1,d2);
			System.out.println(str+"\n");
//			System.exit(0);
			
			for(int i=0;i<n_b;i++)
				d[1][i]=new DataPoint(d2[i],1,i);
			TestResults t=new TestResults("Mann Whittley");
			studentT_Test(t,d,0);
			mannWhitney(t,d);
			robustRankSum(t,d);

			int m=7;
			int n=8;
			double []d3={5.025,6.7,6.725,6.75,7.05,7.25,8.375};
			double[]d4={4.875,5.125,5.225,5.425,5.55,5.75,5.925,6.125};
			d = new DataPoint[2][];
			d[0]=new DataPoint[m];
			d[1]=new DataPoint[n];
			for(int i=0;i<m;i++)
				d[0][i]=new DataPoint(d3[i],0,i);
			for(int i=0;i<n;i++)
				d[1][i]=new DataPoint(d4[i],1,i);
			
			studentT_Test(t,d,0);
			mannWhitney(t,d);
			robustRankSum(t,d);
		
			str = ts.performTests(d3,d4);
			System.out.println(str+"\n");
			System.exit(0);
		
		}

	public static void wilcoxonMatchedPairs(TestResults t, DataPoint[][] d)
	{
		if(d.length!=2)
		{
			System.out.println("Error, cannot use this for k!=2");
			System.exit(0);
		}
		if(d[0].length!=d[1].length)
		{
			System.out.println("Error, cannot use this for unequal samples, they should be matched");
			System.exit(0);
		}
		DataPoint[] ranked= new DataPoint[d[0].length];
		for(int i=0;i<ranked.length;i++)
		{
			System.out.println(" Difference ="+(d[0][i].d-d[1][i].d));
			ranked[i]= new DataPoint(d[0][i].d-d[1][i].d,0,d[0][i].position);
		}	
		Tests.rank(ranked);
		OneSampleTests.wilcoxonSignRank(t,ranked);
}			


	public static void main(String args[]) {
            testTwoSamples();
            System.exit(0);

            String fileName = "C:\\Users\\ajb\\Dropbox\\Results\\DebugFiles\\TwoSampleTest.csv";
            double[][] d;
            TestResults T=new TestResults("SSS");
            T.h0=0;
            T.level=0.05;
            T.type=0;
            loadData(fileName);
            
            
            wilcoxonMatchedPairs(T,dataByLevel);	
            System.out.println(T);
/*		
		d=getData(fileName);
	
		switch(testType)
		{
		//Location
			case 0:	//Independent, normal, unknown variance
				T=T_Test(d);
				break;
				//Independent, unknown distribution		
			case 1: 	
				T=MannWhitney(d);
				break;
			case 2: 	
				T=Wilcoxon(d);
				break;
			case 3: 	
				T=Tukey(d);
				break;
			case 4:	//Dependent, Just do single sample test on di
			
*/
	}

}
