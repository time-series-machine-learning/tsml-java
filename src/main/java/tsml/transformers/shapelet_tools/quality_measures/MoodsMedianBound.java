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
package tsml.transformers.shapelet_tools.quality_measures;
import utilities.class_counts.ClassCounts;
import utilities.class_counts.SimpleClassCounts;
import tsml.transformers.shapelet_tools.OrderLineObj;

    
    /**
     * A class for calculating the moods median statistic bound of a shapelet, according to 
     * the set of distances from the shapelet to a dataset.
     */
    public class MoodsMedianBound extends ShapeletQualityBound{  
        
        /**
         * Constructor to construct MoodsMedianBound
         * @param classDist class distribution of the data currently being processed
         * @param percentage percentage of data required to be processed before
         *                   bounding mechanism is used.
         */
        protected MoodsMedianBound(ClassCounts classDist, int percentage){
            initParentFields(classDist, percentage);
        }
                
        /**
         * Method to calculate the quality bound for the current orderline
         * @return Moods Median statistic bound
         */
        @Override
        protected double calculateBestQuality(){
            int lengthOfOrderline = orderLine.size();
            double median;
            if(lengthOfOrderline%2==0){
                median = (orderLine.get(lengthOfOrderline/2-1).getDistance()+orderLine.get(lengthOfOrderline/2).getDistance())/2;
            }else{
                median = orderLine.get(lengthOfOrderline/2).getDistance();
            }

            int totalCount = orderLine.size();
            int countBelow = 0;
            int countAbove = 0;
            int numClasses = parentClassDist.size();
            
            ClassCounts classCountsBelowMedian = new SimpleClassCounts(numClasses);
            ClassCounts classCountsAboveMedian = new SimpleClassCounts(numClasses);

            double distance;
            double classVal;
            
            // Count observed class distributions above and below the median
            for (OrderLineObj orderLine1 : orderLine) {
                distance = orderLine1.getDistance();
                classVal = orderLine1.getClassVal();
                if(distance < median){
                    countBelow++;
                    classCountsBelowMedian.addTo(classVal, 1); //increment by 1
                }else{
                    countAbove++;
                    classCountsAboveMedian.addTo(classVal, 1);
                }
            }
            
            // Add count of predicted class distributions above and below the median
            for(double key : orderLineClassDist.keySet()){
                int predictedCount = parentClassDist.get(key) - orderLineClassDist.get(key);
                if(classCountsBelowMedian.get(key) <= classCountsAboveMedian.get(key)){
                    classCountsAboveMedian.addTo(key, predictedCount);
                    countAbove += predictedCount;
                }else{
                    classCountsBelowMedian.addTo(key, predictedCount);
                    countBelow += predictedCount;
                }
                totalCount += predictedCount;
            }
            
            double chi = 0;
            double expectedAbove = 0, expectedBelow;
            for(int i = 0; i < numClasses; i++){
                expectedBelow = (double)(countBelow*parentClassDist.get((double)i))/totalCount;
                double classCountsBelow = classCountsBelowMedian.get(i) - expectedBelow;
                double classCountsAbove = classCountsAboveMedian.get(i) - expectedAbove;
                chi += (classCountsBelow*classCountsBelow)/expectedBelow;

                expectedAbove = (double)(countAbove*parentClassDist.get((double)i))/totalCount;
                chi += (classCountsAbove*classCountsAbove)/expectedAbove;
            }

            if(Double.isNaN(chi)){
                chi = 0; // fix for cases where the shapelet is a straight line and chi is calc'd as NaN
            }
            return chi;
        }
        
        @Override
        public boolean pruneCandidate(){
            if(orderLine.size() % parentClassDist.size() != 0){//if(orderLine.size() < parentClassDist.size()){
                return false;
            }else{
                return super.pruneCandidate();
            }
        }    
    }   
    
    

