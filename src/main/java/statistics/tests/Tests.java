package statistics.tests;
import fileIO.*;
import java.util.*;

abstract public class Tests{


	protected static int N;	//Total data size
	protected 	static int k;	//number of levels, default to 1
	protected 	static int[] n;	//number of data per level
	protected 	static DataPoint[][] dataByLevel;
	protected 	static DataPoint[] rankedData;
	protected 	static boolean loaded = false;	
	protected 	static boolean debug = false;	
/*
loadData MUST be called prior to any of the tests

It loads the data from file, sorts it, ranks it and adjusts ranks for duplicates

Implementation notes
1. 	Input must be csv format, c++ type comments are ignored
Data must be in ROWS, i.e. 1 row = i sample. Unusual, but I'm working with row based data!
Data is Stored row based (i == sample index, j== data index)
2. 	First implementation assumes equal sample sizes at each level
3. 	First line of data should contain the number of treatment levels
4. 	Second line should store the number of data in each sample

*/	
	public static void loadData(String fileName)
	{
		InFile f	= new InFile(fileName);
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
		Arrays.sort(rankedData);
		for(int i=0;i<N;i++)
			rankedData[i].rank=(i+1);
		adjustRanksForDuplicates(rankedData);
		loaded =true;
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
	public static void loadData(double[][] data)
	{
		k = data.length;
		n = new int[k];
		N=0;
		dataByLevel = new DataPoint[k][];
		for(int i=0;i<k;i++)
		{
			n[i]=data[i].length;
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
				dataByLevel[i][j]=new DataPoint(data[i][j],i,j);
				rankedData[c]=dataByLevel[i][j];
				c++;
			}	
		}
		Arrays.sort(rankedData);
		for(int i=0;i<N;i++)
			rankedData[i].rank=(i+1);
		adjustRanksForDuplicates(rankedData);
		loaded =true;
	}
	public static void loadData(double[] data)
// Only use with OneSample Tests
	{
		k = 1;
		n = new int[k];
		N=data.length;
		n[0]=data.length;
		dataByLevel = new DataPoint[k][];
		
		double d;
		int c=0;
		rankedData = new DataPoint[N];
		for(int j=0;j<N;j++)
		{
			dataByLevel[0][j]=new DataPoint(data[j],0,j);
			rankedData[j]=dataByLevel[0][j];
		}	
		Arrays.sort(rankedData);
		for(int i=0;i<N;i++)
			rankedData[i].rank=(i+1);
		adjustRanksForDuplicates(rankedData);
		loaded =true;
	}

	protected static void	adjustRanksForDuplicates(DataPoint[] ranks)
	{
		DataPoint first=ranks[0];
		int count=0;
		int pos=0;
		double s,e;
		for(int i=1;i<ranks.length;i++)
		{
			if(ranks[i].d!=first.d)
			{
				if(count>0)
				{
					s=first.rank;
					e=i;
//					System.out.print("First Rank = "+s+"\nLast Rank ="+e+"\n count = "+count+"\n");

					for(int j=0;j<=count;j++)
						ranks[(int)(s-1)+j].rank=(e+s)/2;
					count=0;	
				}
				first=ranks[i];
			}
			else
				count++;
		}
		if(count>0)
		{			
			s=first.rank;
			e=ranks.length;
//			System.out.print("First Rank = "+s+"\nLast Rank ="+e+"\n count = "+count+"\n");

			for(int j=0;j<=count;j++)
				ranks[(int)(s-1)+j].rank=(e+s)/2;
			count=0;	
		}			
	}
	public static void rank(DataPoint[] data)
	{
		Arrays.sort(data);
		for(int i=0;i<data.length;i++)
			data[i].rank=(i+1);
		adjustRanksForDuplicates(data);

	}		
}

