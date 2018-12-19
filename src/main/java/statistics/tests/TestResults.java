package statistics.tests;

import statistics.distributions.*;

public class TestResults{
	public double testStat;
	public double pValue;
	public String testName;
	public double h0;
	public int type;
	//if type ==-1 then H1 \phi < \phi0 
	//if type ==0 then H1 \phi \neq \phi0 (2-Sided)
	//if type ==1 then H1 \phi > \phi0 
	public double level=0.05;
	public double criticalValue;
	public int df1,df2;	//Degrees of freedom, if required
	
//Distribution of test stat if available
//NOTE that all the Distribution classes should be 
//checked before useage, cos at least one of them was wrong!

	public Distribution dist;
	public TestResults(String s){
		testName = s;
		dist = null;
		testStat=pValue=h0=0;
	}	
	public TestResults(String s,Distribution d){
		testName = s;
		dist = d;
		testStat=pValue=h0=0;
	}
	public void findPValue(){
		if(dist==null)
			return;

		pValue=dist.getCDF(testStat);
		if(type==1)
			pValue=1-pValue;
		else if(type==0)
		{
			if(pValue>0.5)
				pValue=1-pValue;
			pValue*=2;
		}			
	}
	public void findCriticalValue()
	{
		if(type==-1)
			criticalValue=dist.getQuantile(1-level);
		else
			criticalValue=dist.getQuantile(level);
	}		
	public String toString(){
            String str="****** Results for "+testName+"   *********\n";
            str+=" To test median ="+h0+" against H1: ";
            if(type==-1)
                    str+="median < "+h0;
            else if(type==1)
                    str+="median > "+h0;
            else
                    str+="median not equal to "+h0;

            str+="\n T = "+testStat+"\t p value = "+pValue+"\tlevel = "+level+"\n";
            str+=" Distribution = "+dist+"\t Level ="+level+"\t Critical Value ="+criticalValue;				
            return str;
	}	
}		
	
