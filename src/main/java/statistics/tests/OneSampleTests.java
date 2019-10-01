/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package statistics.tests;
/* To Test whether population median is zero or not

USAGE:
        OneSampleTests test=new OneSampleTests();
        data: double[] data;
    String str=test.performTests(data);
OUTPUT: A comma separated string with testName,testStatistic,pValue (assuming alpha 0.05)


*/
import statistics.distributions.NormalDistribution;
import statistics.distributions.BinomialDistribution;
import java.util.*;
import fileIO.*;
import java.text.DecimalFormat;
import statistics.distributions.StudentDistribution;
import statistics.tests.DataPoint;
import statistics.tests.TestResults;
import statistics.tests.Tests;

public class OneSampleTests extends Tests{
    public static boolean beQuiet = false;
    
    private static DecimalFormat df = new DecimalFormat("##.########");
    public static DataPoint[] absRankedData;

    public String performTests(double[] data){
        DataPoint[] d=new DataPoint[data.length];
        boolean allthesame=true;
        for(int i=0;i<data.length;i++){
            d[i]=new DataPoint(data[i],0,i);
            if(allthesame && data[i]!=data[0])
                allthesame=false;
        }
        if(allthesame){
            System.out.println("All the values are identical, not performing tests");
            return "T_Test,0,0.5,Sign_Test,0,0.5,Sign_Rank_Test,0,0.5";
        }
        Arrays.sort(d);
        TestResults test=new TestResults("T_Test");
        studentTTest(test,d);
        test.findPValue();
        String str="T_Test,"+test.testStat+","+df.format(test.pValue)+",";
        test=new TestResults("SignTest");
        signTest(test,d);
        test.findPValue();
        str+="Sign_Test,"+test.testStat+","+df.format(test.pValue)+",";
        test.findPValue();
        test=new TestResults("WilcoxonSignRankTest");
        wilcoxonSignRank(test,d);
        test.findPValue();
        str+="Sign_Rank_Test,"+test.testStat+","+df.format(test.pValue);
        return str;
    }
    public static void studentTTest(TestResults t, DataPoint[] data){
//Find mean and var. Var found the slow way for clarity
        double mean=0,var=0;
        for(int i=0;i<data.length;i++)
            mean+=data[i].d;
        mean/=data.length;
        for(int i=0;i<data.length;i++)
            var+=(mean-data[i].d)*(mean-data[i].d);
        var/=data.length-1;
//Find test stat
        double tStat=(mean)/Math.sqrt(var/data.length);
	t.testStat=tStat;		
        t.df1=data.length-1;
        t.dist=new StudentDistribution(t.df1);
        
    }

    public static void signTest(TestResults T, DataPoint[] ranked)
/* 	Defined by J. Arbuthnott in 1710!
            s1=nos greater than \phi_0
            s2=nos less than \phi_0
            Ties are shared equally	

PRE: ASSUMES Data pre sorted
    Note this does not require sorted data, so its a bit 
    wasteful using the loadData if signTest is being performed.
    BUT given it is presorted, we can speed up 
    the calculation with a binary search
It sets the distribution of the test statistic	
*/
    {
        T.testName="signTest";	
        double s1=0,s2=0;
        DataPoint h0 = new DataPoint(T.h0,0,0);
        int adjN=ranked.length;
        int pos=Arrays.binarySearch(ranked,h0);

/* From API
index of the search key, if it is contained in the list; 
otherwise, (-(insertion point) - 1). The insertion point is 
defined as the point at which the key would be inserted into 
the list: the index of the first element greater than the key, 
or list.size(), if all elements in the list are less than the #
specified key. Note that this guarantees that the return value
will be >= 0 if and only if the key is found.

Tested with file signTestExample.txt
*/

        if(pos>=0) 
//Value present, need to adjust for duplicates
        {
            int dupCount=1;
            int left=pos-1;
            int right=pos+1;
            while(left>=0 && ranked[pos].equals(ranked[left]))
            {
//				System.out.println("Left ="+ranked[left].d);
                left--; 
                dupCount++;
            }			
            while(right< ranked.length && ranked[pos].equals(ranked[right]))
            {
                right++; 
                dupCount++;
            }
            if(dupCount%2==1&& adjN<50)	//If using a binomial want a whole number!
            {
                adjN--;
                dupCount-=1;
            }
//            System.out.println("Duplicate count ="+dupCount);
            s1=left+1+dupCount/2.0;
            s2=(ranked.length-right)+dupCount/2.0;			
//            System.out.println("left = "+left+"\t right = "+right+"\t lower ="+s1+"\thigher = "+s2);

        }
        else
        {
    //Number smaller
            s1=-pos-1;
    //Number larger: THIS MAY BE A BUG IF adjN is adjusted
            s2=adjN-s2;
//            System.out.println("pos = "+pos+"\tlower ="+s1+"\thigher = "+s2);
        }	

        if(T.type==-1)
            T.testStat=s2;
        else if(T.type==1)
            T.testStat=s1;
        else if(T.type==0)
            T.testStat=(s1<s2)?s1:s2;

//Test distribution, use binomial approximation if N>50
        if(adjN<50)
            T.dist = new BinomialDistribution(adjN,0.5);
        else
            T.dist = new NormalDistribution(adjN/2.0,Math.sqrt(adjN)/2.0);
        T.findCriticalValue();
        T.findPValue();		
    }

    public static void wilcoxonSignRank(TestResults T, DataPoint[] ranked)
/*
1. Find the differences from hypothesised median
2. Rank by absolute values
3. Sum ranks for positive 
Need to rerank the data

*/
    {
        T.testName="wilcoxonSignRank";	
        absRankedData=new DataPoint[ranked.length];
        double diff;
        int nonZeroDifferences=0;
        for(int j=0;j<ranked.length;j++)
        {
//            System.out.println(" Data = "+ranked[j].d+"\t in Pos "+ranked[j].position);//+" data ="+dataByLevel[0][ranked[j].position].d);
            diff=(ranked[j].d>T.h0)?ranked[j].d-T.h0:T.h0-ranked[j].d;
            if(diff>0){
                absRankedData[nonZeroDifferences]=new DataPoint(diff,0,j);
                nonZeroDifferences++;
            }	
        }
        if(nonZeroDifferences==0){
            System.out.println(" The two series are identical, this should have been handled before this call to Wilcoxon Sign Rank test. Setting up test to accept the null");
            T.dist = new NormalDistribution(0,1);
             T.testStat=0;
            T.findCriticalValue();
            T.findPValue();		


            return;  
            
        }
        if(nonZeroDifferences<ranked.length){
            DataPoint[] temp = new DataPoint[nonZeroDifferences];
            for(int i=0;i<nonZeroDifferences;i++)
                    temp[i]=absRankedData[i];
            absRankedData=temp;	
        }	
        Arrays.sort(absRankedData);
        for(int i=0;i<absRankedData.length;i++)
                absRankedData[i].rank=(i+1);
        adjustRanksForDuplicates(absRankedData);

        double rankSumUnder=0,rankSumOver=0;

        for(int j=0;j<nonZeroDifferences;j++)
        {
            diff=ranked[absRankedData[j].position].d-T.h0;
            if (!beQuiet)
                System.out.println(" Rank = "+ j +" Pos ="+absRankedData[j].position+" Val ="+ranked[absRankedData[j].position].d+" diff ="+diff+" Abs Val ="+absRankedData[j].d);
            if(diff<0)
            {
                if (!beQuiet)
                    System.out.println(" Rank = "+ absRankedData[j].rank +" Value ="+ranked[absRankedData[j].position].d+"\t diff ="+diff);
                rankSumUnder+=absRankedData[j].rank;
            }	
            else	
                rankSumOver+=absRankedData[j].rank;
        }
        if(T.type==1)
            T.testStat=rankSumUnder;
        else if(T.type==-1)
            T.testStat=rankSumOver;
        else
            T.testStat=(rankSumOver<rankSumUnder)?rankSumUnder:rankSumOver;
        //Havent used the exact distribution, but it is possible for
        //small N, see page 74 of Neave	
        T.dist = new NormalDistribution(nonZeroDifferences*(nonZeroDifferences+1)/4.0,Math.sqrt(nonZeroDifferences*(nonZeroDifferences+1)*(2*nonZeroDifferences+1)/24.0));	
        T.findCriticalValue();
        T.findPValue();		

    }

    public static void main(String[] args){
            TestResults t = new TestResults("SignTest");

            InFile inf=new InFile("C:\\Users\\ajb\\Dropbox\\Results\\DebugFiles\\TwoSampleTest.csv");				
            int n=inf.readInt();
            int m=inf.readInt();
            double[] diff=new double[m];
            for (int i = 0; i < diff.length; i++)
                diff[i]=inf.readDouble();
            for (int i = 0; i < diff.length; i++)
                diff[i]-=inf.readDouble();
            OneSampleTests one= new OneSampleTests();
            System.out.println(one.performTests(diff));
    //Sign Test

    }
}
