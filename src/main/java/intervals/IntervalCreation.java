/*
 * Copyright (C) 2019 xmw13bzu
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

package intervals;

import timeseriesweka.filters.NormalizeCase;
import weka.core.Instances;

/**
 * Todo refactor into a weka filter/ueatsc transformer at some point 
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public class IntervalCreation {
    
    
    
    public static Instances crop_proportional(Instances insts, double startProp, double endProp, boolean normalise) throws Exception { 
        int startInd = (int) ((insts.numAttributes()-1) * startProp);
        int endInd = (int) ((insts.numAttributes()-1) * endProp);

        return crop_proportional(insts, startInd, endInd, normalise);
    }
    
    
    //maybe theres a filter that does what i want?... faster to just do it here
    public static Instances crop_proportional(Instances insts, int startInd, int endInd, boolean normalise) throws Exception { 
//        System.out.println(insts.numAttributes());
//        System.out.println(insts.numInstances());
//        System.out.println(insts.numClasses());

//        System.out.println("");
//        System.out.println("inteveral = " + startProp + " - " + endProp);
//        System.out.println("startTimePoint = " + startTimePoint);
//        System.out.println("endTimePoint = " + endTimePoint);
//        System.out.println("");
        
        if (startInd == endInd) {//pathological case for very short series
            if (endInd < (insts.numAttributes()-2))
                endInd++;
            else if (startInd > 0)
                startInd--;
            else 
                throw new Exception("Interval wont work, " + insts.relationName() + " startind=" + startInd + " endind=" + endInd + " numAtts=" + insts.numAttributes());
        }
        
        Instances cropped = new Instances(insts);

        for (int ind = cropped.numAttributes()-2; ind >= 0; ind--)
            if (ind < startInd || ind > endInd)
                cropped.deleteAttributeAt(ind);
        
//        System.out.println(cropped.numAttributes());
//        System.out.println(cropped.numInstances());
//        System.out.println(cropped.numClasses());
        
        if (normalise)
            return (new NormalizeCase()).process(cropped);
        else
            return cropped;
    }
}
