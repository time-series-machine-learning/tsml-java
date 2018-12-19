package statistics.tests;
/*
Kruscal-Wallis non-parametric test for difference of k-means

**For description used in this implementation, see Neave and Worthington
Distribution-Free Tests**

K-W test is equivalent to ANOVA f-test, except based on RANK rather than observed value.
Not analgous to the way Spearman's rank correlation coefficient is calculated.
K-W is a generalisation of the Mann-Whitney test for difference of two means
Assumes sample independence.
Assumes continuous distribution (no ties), although this is not essential (see below)

H_0: There is no difference is population means the samples were taken from
H_1: There are some differences

Test stat calculated thus

1. Find the rank of each data in terms of all samples pooled together
(For ties use the average position and adjustment described below if there are a lot of ties).
2. For each sample calculate the average ranks in the sample R_i and the overall mean
R
3. Find the weighted sum of the squared deviations from the mean (weighted by each sample size) W.
4. Calculate test stat
H=W*12/(N*(N+1))
where N is the total number of data, or more easily

H = [12/N(N+1)]* sum{R_i/n_i} - 3(N+1)
5. Large value of H are unlikely if H_0 is true. Not sure of the ddistribution of H under
H_0, probably ML test. Critical regions from tables

If there are a large number of ties an adjustment should be made. Suppose r values have more than
one occurence, then H should be divided by C below. Let t_i be the number of ties for a given data
value, then 
C = 1 - sum{t^3 - t}/[N(N^2-1)]
H*=H/C

Implementation notes
1. 	Input must be csv format, c++ type comments are ignored
Data must be in ROWS, i.e. 1 row = i sample
2. 	First implementation assumes equal sample sizes at each level
3. 	First line of data should contain the number of treatment levels
4. 	Second line should store 
4. 	The file Test1.csv contains the example from page 245 of Neive
	The file Test2.csv contains the example from page 249 of Neive that has duplications
5. 	Output 

*/
import fileIO.*;
import java.util.Arrays;
public class KruskalWallis {

	static InFile f;
	static int N;	//Total data size
	static int k;	//number of levels
	static int[] n;	//number of data per level
	static String fileName = "LearningRate100Rules.csv";	//Hack,. read in from args
//	static String fileName = "Test2.txt";	//Hack,. read in from args
	static DataPoint[][] dataByLevel;
	static DataPoint[] rankedData;
	static boolean debug = false;	
	
	public static void main(String args[]) {
		
		double H;
		double C;
		double H_prime;
		
		loadData();
		System.out.println("FILE ="+fileName+"\n Treatment levels ="+k+"\t Total data = "+N+" per level ="+n[0]);
//Sort Data		
		Arrays.sort(rankedData);

		for(int i=0;i<N;i++)
		{
			rankedData[i].rank=(i+1);
			if(debug)
				System.out.print(rankedData[i].d+"\t");
		}		
//Check for duplicates, counts number duplicated, and then recalculates ranks as the averages

//		System.out.print("\n\nPRIOR TO Duplicate\n");
//		for(int i=0;i<N;i++)
//			System.out.print(rankedData[i].rank+"\t");
		adjustRanksForDuplicates();
	
		
//Find rank sums			
		double[] rankSums= new double[k];
		for(int i=0;i<k;i++)
		{
			rankSums[i]=0;
			for(int j=0;j<n[i];j++)
				rankSums[i]+=dataByLevel[i][j].rank;				
		}
// Find H stat

		H=0;
		for(int i=0;i<k;i++)
			H+=rankSums[i]*rankSums[i]/n[i];
		H=H*12/(N*(N+1));
		H-=3*(N+1);
		System.out.println("\n\n H stat = "+H);									

//Find C stat
		int i=0;
		int nextPos=0;
		int t=0;
		int tSum=0,t3Sum=0;
		while(i<N)
		{
			if(rankedData[i].rank!=(i+1))
			{
				t=(int)(2*rankedData[i].rank)-i;
//				System.out.println("\n t = "+t+"i = "+i); 
				nextPos = t-1;
				t= (t-i-1);
				tSum+=(t);
				t3Sum+=t*t*t;
				i=nextPos;
			}
			else
				i++;
		}

		System.out.println("\n\n t sum = "+tSum+"\t t^3 sum = "+t3Sum);									
		C=1- ((double)t3Sum-tSum)/(N*(N*N-1));
		H_prime = H/C;
		System.out.println("\n\n C = "+C+"\t H* = "+H_prime);									
								
	}
	public static void loadData()
	{
		f	= new InFile(fileName);
		k = f.readInt();
		n = new int[k];
		N=0;
		dataByLevel = new DataPoint[k][];
		for(int i=0;i<k;i++)
		{
			n[i]=f.readInt();
			N+=n[i];
			dataByLevel[i]=new DataPoint[n[i]];
		}
		
		double d;
		int c=0;
		rankedData = new DataPoint[N];
		for(int i=0;i<k;i++)
		{
			for(int j=0;j<n[i];j++)
			{
				d = f.readDouble();
				dataByLevel[i][j]=new DataPoint(d,i,j);
				rankedData[c]=dataByLevel[i][j];
				c++;
			}	
		}
		if(debug)
		{
			for(int i=0;i<k;i++)
			{
				for(int j=0;j<n[i];j++)
					System.out.print(dataByLevel[i][j].d+"\t");
				System.out.print("\n");
			}	
		}		
	}
	public static void	adjustRanksForDuplicates()
	{
		DataPoint first=rankedData[0];
		int count=0;
		int pos=0;
		double s,e;
		for(int i=1;i<N;i++)
		{
			if(rankedData[i].d!=first.d)
			{
				if(count>0)
				{
					s=first.rank;
					e=i;
//					System.out.print("First Rank = "+s+"\nLast Rank ="+e+"\n count = "+count+"\n");

					for(int j=0;j<=count;j++)
						rankedData[(int)(s-1)+j].rank=(e+s)/2;
					count=0;	
				}
				first=rankedData[i];
			}
			else
				count++;
		}
		if(count>0)
		{			
			s=first.rank;
			e=N;
//			System.out.print("First Rank = "+s+"\nLast Rank ="+e+"\n count = "+count+"\n");

			for(int j=0;j<=count;j++)
				rankedData[(int)(s-1)+j].rank=(e+s)/2;
			count=0;	
		}			
	}
		
}

