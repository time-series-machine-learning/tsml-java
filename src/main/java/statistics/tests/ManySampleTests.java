package statistics.tests;
import fileIO.*;
import java.util.Arrays;

public class ManySampleTests extends Tests{

	


/*
Kruscal-Wallis non-parametric test for difference of k-means

**For description used in this implementation, see Neave and Worthington
Distribution-Free Tests**

K-W test is equivalent to ANOVA f-test, except based on RANK rather than observed value.
Analagous to the way Spearman's rank correlation coefficient is calculated.
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
5. Large value of H are unlikely if H_0 is true. Not sure of the distribution of H under
H_0, probably ML test. Critical regions from tables

If there are a large number of ties an adjustment should be made. Suppose r values have more than
one occurence, then H should be divided by C below. Let t_i be the number of ties for a given data
value, then 
C = 1 - sum{t^3 - t}/[N(N^2-1)]
H*=H/C

The file Test1.csv contains the example from page 245 of Neive
The file Test2.csv contains the example from page 249 of Neive that has duplications

ASSUMES
1. Data loaded
2. Data ranked and sorted 
3. Ranks adjusted for duplicates (2 and 3 are done in the loading methods)
*/

	public static void kruskalWallace(TestResults T)
	
	{
		if(!loaded)
		{
			System.out.println("ERROR: Data not loaded, cannot perform testg");
			return;
		}
		double H;
		double C;
		double H_prime;
		T.testName="Kruskal Wallace";
				
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
		T.testStat=H_prime;
								
	}

	public static void main(String[] args)
	{
		String fileName = "C:/JavaSource/Clustering/Clustering_Data/Java_Application_Release/Experiment3/Exp3NoRandomShocks.txt";

		loadData(fileName);
		TestResults t =new TestResults("Blank");
		t.h0=0;
		t.type=0;
		kruskalWallace(t);
		System.out.println(" H prime = "+t.testStat);
	}		

}

