/*
 * This file is part of the UEA Time Series Machine Learning (TSML) toolbox.
 *
 * The UEA TSML toolbox is free software: you can redistribute it and/or 
 * modify it under the terms of the GNU General Public License as published 
 * by the Free Software Foundation, either version 3 of the License, or 
 * (at your option) any later version.
 *
 * The UEA TSML toolbox is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with the UEA TSML toolbox. If not, see <https://www.gnu.org/licenses/>.
 */
 

/*
 * Created on Jan 30, 2006
 *
 * TODO To change the template for this generated file go to
 * Window - Preferences - Java - Code Style - Code Templates
 */
package statistics.transformations;

import java.io.FileReader;
import fileIO.*;
import weka.core.*;
import weka.attributeSelection.PrincipalComponents;

public class PowerSearch {

	public static double MIN=-4, MAX=4, INCREMENT=0.125;
//First rows is all ones
//Last row is for transformed attribute	
	public static double transformRegressor(double[][] data, double[] response, int pos)
	{

//1. Get values of the attribute of interest. 
		double[] originalData= new double[data[pos].length];
		double[] transformedData= new double[originalData.length];

		for(int i=0;i<originalData.length;i++)
		{
			originalData[i]=data[pos][i];
//			if(i<10)
//				System.out.println(" data "+i+" = "+data[pos][i]+" response = "+response[i]);
		}
		data[pos]=transformedData;
		LinearModel l;
		double alpha, s,minAlpha=0,minSSE=Double.MAX_VALUE;
		for(alpha=MIN;alpha<=MAX;alpha+=INCREMENT)
		{
			if(alpha==0)
			{
				for(int j=0;j<originalData.length;j++)
					transformedData[j]=Math.log(originalData[j]);
			}
			else
			{
				for(int j=0;j<originalData.length;j++)
					transformedData[j]=Math.pow(originalData[j],alpha);
			}
			l=new LinearModel(data,response);
			l.fitModel();
			s=l.findStats();
//			System.out.println(" Alpha = "+alpha+" SSE = "+s);
			if(s<minSSE)
			{
				minAlpha=alpha;
				minSSE=s;
			}
		}
		if(minAlpha==MIN || minAlpha==MAX)
			minAlpha=1;
		return minAlpha;
	}
	
	public static double[] transform(double[] x, double power)
	{
		double[] newX= new double[x.length];
		for(int i=0;i<x.length;i++)
			newX[i]=Math.pow(x[i],power);
		return newX;
	}
	public static void main(String[] args)
	{
		Instances data=null;
		try{
			FileReader r = new FileReader("C:/Research/Code/Archive Generator/src/weka/addOns/BoxTidwellTest2.arff");
			data = new Instances(r);
			data.setClassIndex(data.numAttributes()-1);
		}catch(Exception e)
		{
			System.out.println("Error loading file "+e);
		}
		double[] powers=new double[data.numAttributes()-1];
//		data=transformRegressor(data,0,powers);
//		data=transformRegressor(data,2,powers);
//		data=transformRegressor(data,1,powers);

		System.out.println(" Final powers =");
		for(int i=0;i<powers.length;i++)
			System.out.println(i+" ="+powers[i]);
			
		OutFile r = new OutFile("C:/Research/Code/Archive Generator/src/weka/addOns/BoxTidwellResults2.arff");
		r.writeLine(data.toString());
		
	}
	
}
