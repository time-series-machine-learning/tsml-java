/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package timeseriesweka.filters.shapelet_transforms.quality_measures;
import utilities.class_distributions.ClassDistribution;
import utilities.class_distributions.SimpleClassDistribution;
import timeseriesweka.filters.shapelet_transforms.OrderLineObj;

    
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
        protected MoodsMedianBound(ClassDistribution classDist, int percentage){
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
            
            ClassDistribution classCountsBelowMedian = new SimpleClassDistribution(numClasses);
            ClassDistribution classCountsAboveMedian = new SimpleClassDistribution(numClasses);

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
    
    

