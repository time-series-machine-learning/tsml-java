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
package timeseriesweka.examples;

import weka.core.Instances;
import weka.filters.Filter;
import timeseriesweka.filters.ACF;
import timeseriesweka.filters.PowerSpectrum;

/**
 *
 * @author ajb
 */
public class TransformExamples {
    public static Instances acfTransform(Instances data){
        ACF acf=new ACF();
	acf.setMaxLag(data.numAttributes()/4);
        Instances acfTrans=null;
        try{
            acf.setInputFormat(data);
            acfTrans=Filter.useFilter(data, acf);
        }catch(Exception e){
                System.out.println(" Exception in ACF harness="+e);
		e.printStackTrace();
               System.exit(0);
        }
      
            return acfTrans;
    }
    public static Instances psTransform(Instances data){
        PowerSpectrum ps=new PowerSpectrum();
        Instances psTrans=null;
        try{
            ps.setInputFormat(data);
            psTrans=Filter.useFilter(data, ps);
            ps.truncate(psTrans, data.numAttributes()/4);
        }catch(Exception e){
                System.out.println(" Exception in ACF harness="+e);
		e.printStackTrace();
               System.exit(0);
        }
           return psTrans;
    }
    
    
}
