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
package timeseriesweka.filters;

import java.io.FileReader;
import java.util.*;
import weka.core.*;
import weka.filters.SimpleBatchFilter;

/*
     * copyright: Anthony Bagnall
 * 
 * */
public class Clipping extends SimpleBatchFilter {
	boolean useMean=true;
	boolean useRealAttributes=false;
	public void setUseRealAttributes(boolean f){useRealAttributes=f;}
	@Override
	protected Instances determineOutputFormat(Instances inputFormat)
			throws Exception {
//Must convert all attributes to binary.
		Attribute a;
		FastVector fv=new FastVector();
		if(!useRealAttributes){
			fv.addElement("0");
			fv.addElement("1");
		}
		FastVector atts=new FastVector();

		
		  for(int i=0;i<inputFormat.numAttributes();i++)
		  {
//			  System.out.println(" Create Attribute "+i);
			  if(i!=inputFormat.classIndex()){
				  if(!useRealAttributes)
					  a=new Attribute("Clipped"+inputFormat.attribute(i).name(),fv);
				  else
					  a=new Attribute("Clipped"+inputFormat.attribute(i).name());
			  }
			  else
				  a=inputFormat.attribute(i);
			  atts.addElement(a);
//			  System.out.println(" Add Attribute "+i);
//			  result.insertAttributeAt(a,i);
		  }
		Instances result = new Instances("Clipped"+inputFormat.relationName(),atts,inputFormat.numInstances());
//		  System.out.println(" Output format ="+result);
                if(inputFormat.classIndex()>=0){
                        result.setClassIndex(result.numAttributes()-1);
                }
                return result;
	}

	@Override
	public String globalInfo() {
		return null;
	}
//Means by CASE, not by attribute	
	private double[] findMedians(Instances instances){
		//USe quick select to find them
		return null;
	}
	private double[] findMeans(Instances instances){
		double[] means=new  double[instances.numInstances()];
		int count=0;
		for(int i=0;i<instances.numInstances();i++){
			count=0;
			for(int j=0;j<instances.numAttributes();j++){
				if(j!=instances.classIndex()&& !instances.instance(i).isMissing(j)){
						count++;
						means[i]+=instances.instance(i).value(j);
					}
			}
			if(count>0) 
				means[i]/=count;
//			System.out.println(" Mean attribute "+j+" = "+means[j]);
		}
		return means;
	}
	@Override
	public Instances process(Instances instances) throws Exception {
		//find the average values, either mean or median
		double[] averages;
		if(useMean)
			averages=findMeans(instances);
		else
			averages=findMedians(instances);
			
		
		  Instances result = determineOutputFormat(instances);
		  Instance newInst;
		  String val="0";
		  if(!useRealAttributes){
			  for(int i=0;i<instances.numInstances();i++) {
				  newInst= new DenseInstance(result.numAttributes());
				  result.add(newInst);
				  for(int j=0;j<instances.numAttributes();j++){
					  if(instances.instance(i).isMissing(j))
						  val="?";
					  else{
						  if(j!=instances.classIndex()){
							  if(instances.instance(i).value(j)<averages[i])	// Zero
								  val="0";
							  else
								  val="1";
							  result.instance(i).setValue(j,val);
						  }
						  else
							  result.instance(i).setValue(j,instances.instance(i).stringValue(j));					  
					  }
				  }  
			  }
		  }
		  else{
			  double x=0;
			  for(int i=0;i<instances.numInstances();i++) {
				  newInst= new DenseInstance(result.numAttributes());
				  result.add(newInst);
				  for(int j=0;j<instances.numAttributes();j++){
					  if(instances.instance(i).isMissing(j))
						  x=-1;
					  else{
						  if(j!=instances.classIndex()){
							  if(instances.instance(i).value(j)<averages[i])	// Zero
								  x=0;
							  else
								  x=1;
							  result.instance(i).setValue(j,x);
						  }
						  else
							  result.instance(i).setValue(j,instances.instance(i).value(j));					  
					  }
				  }  
			  }
		  }

                  return result;
	}

	public String getRevision() {
		return null;
	}
	public static void main(String[] args){
		Clipping cp=new Clipping();
		Instances data=null;
		String fileName="C:\\Research\\Data\\Time Series Data\\Time Series Classification\\TestData\\TimeSeries_Train.arff";
		try{
			FileReader r;
			r= new FileReader(fileName); 
			data = new Instances(r); 

			data.setClassIndex(data.numAttributes()-1);
			
			System.out.println(" Class type numeric ="+data.attribute(data.numAttributes()-1).isNumeric());
			System.out.println(" Class type nominal ="+data.attribute(data.numAttributes()-1).isNominal());

			Instances newInst=cp.process(data);
			System.out.println(newInst);
		}catch(Exception e)
		{
			System.out.println(" Error ="+e);
			StackTraceElement [] st=e.getStackTrace();
			for(int i=st.length-1;i>=0;i--)
				System.out.println(st[i]);
				
		}
	}
}
