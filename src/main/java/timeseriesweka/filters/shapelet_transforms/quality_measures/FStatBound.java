/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package timeseriesweka.filters.shapelet_transforms.quality_measures;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import utilities.class_distributions.ClassDistribution;
import timeseriesweka.filters.shapelet_transforms.OrderLineObj;


/**
 *
 * @author raj09hxu
 */
/**
     * A class for calculating the f-stat statistic bound of a shapelet, according to 
     * the set of distances from the shapelet to a dataset.
     */
    public class FStatBound extends ShapeletQualityBound{  
        
        private double[] sums;        
        private double[] sumsSquared;
        private double[] sumOfSquares;
        private List<OrderLineObj> meanDistOrderLine;
       
        private double minDistance;
        private double maxDistance;
        
        /**
         * Constructor to construct FStatBound
         * @param classDist class distribution of the data currently being processed
         * @param percentage percentage of data required to be processed before
         *                   bounding mechanism is used.
         */
        protected FStatBound(ClassDistribution classDist, int percentage){
            initParentFields(classDist, percentage);
          
            int numClasses = parentClassDist.size();
            sums = new double[numClasses];
            sumsSquared = new double[numClasses];
            sumOfSquares = new double[numClasses];
            meanDistOrderLine = new ArrayList<>(numClasses);
            minDistance = -1.0;
            maxDistance = -1.0;
        }
        
        @Override
        public void updateOrderLine(OrderLineObj orderLineObj){
            super.updateOrderLine(orderLineObj);
            
            int c = (int) orderLineObj.getClassVal();
            double thisDist = orderLineObj.getDistance();
            sums[c] += thisDist;
            sumOfSquares[c] += thisDist * thisDist;
            sumsSquared[c] = sums[c] * sums[c];
            
            //Update min/max distance observed so far
            if(orderLineObj.getDistance() != 0.0){
                if(minDistance == -1 || minDistance > orderLineObj.getDistance()){
                    minDistance = orderLineObj.getDistance();
                }
            
                if(maxDistance == -1 || maxDistance < orderLineObj.getDistance()){
                    maxDistance = orderLineObj.getDistance();
                }
            }
            
            //Update mean distance orderline
            boolean isUpdated = false;
            for (OrderLineObj meanDistOrderLine1 : meanDistOrderLine) {
                if (meanDistOrderLine1.getClassVal() == orderLineObj.getClassVal()) {
                    meanDistOrderLine1.setDistance(sums[(int)orderLineObj.getClassVal()] / orderLineClassDist.get(orderLineObj.getClassVal()));
                    isUpdated = true;
                    break;
                }
            }
            
            if(!isUpdated){
                meanDistOrderLine.add(new OrderLineObj(sums[(int)orderLineObj.getClassVal()] / orderLineClassDist.get(orderLineObj.getClassVal()), orderLineObj.getClassVal()));
            }
        }

        
        /**
         * Method to calculate the quality bound for the current orderline
         * @return F-stat statistic bound
         */
        @Override
        public double calculateBestQuality() {
            int numClasses = parentClassDist.size();
            
            //Sort the mean distance orderline
            Collections.sort(meanDistOrderLine);
            
            //Find approximate minimum orderline objects
            OrderLineObj min = new OrderLineObj(-1.0, 0.0);
            for(Double d : parentClassDist.keySet()){
                int unassignedObjs = parentClassDist.get(d) - orderLineClassDist.get(d);
                double distMin = (sums[d.intValue()] + (unassignedObjs * minDistance)) / parentClassDist.get(d);
                if(min.getDistance() == -1.0 || distMin < min.getDistance()){
                    min.setDistance(distMin);
                    min.setClassVal(d);
                }
            }
            
            //Find approximate maximum orderline objects
            OrderLineObj max = new OrderLineObj(-1.0, 0.0);
            for(Double d : parentClassDist.keySet()){
                int unassignedObjs = parentClassDist.get(d) - orderLineClassDist.get(d);
                double distMax = (sums[d.intValue()] + (unassignedObjs * maxDistance)) / parentClassDist.get(d); 
                if(d != min.getClassVal() && (max.getDistance() == -1.0 || distMax > max.getDistance())){
                    max.setDistance(distMax);
                    max.setClassVal(d);
                }
            }
            
            //Adjust running sums
            double increment = (max.getDistance() - min.getDistance()) / (numClasses-1);
            int multiplyer = 1;
            for (OrderLineObj currentObj : meanDistOrderLine) {
                double thisDist;
                int unassignedObjs = parentClassDist.get(currentObj.getClassVal()) - orderLineClassDist.get(currentObj.getClassVal());
                
                if(currentObj.getClassVal() == min.getClassVal()){
                    thisDist = minDistance;
                }else if(currentObj.getClassVal() == max.getClassVal()){
                    thisDist = maxDistance;
                }else{
                    thisDist = minDistance + (increment * multiplyer);
                    multiplyer++;        
                }
                sums[(int)currentObj.getClassVal()] += thisDist * unassignedObjs;
                sumOfSquares[(int)currentObj.getClassVal()] += thisDist * thisDist * unassignedObjs;
                sumsSquared[(int)currentObj.getClassVal()] = sums[(int)currentObj.getClassVal()] * sums[(int)currentObj.getClassVal()];
            }
            
            double ssTotal;
            double part1 = 0;
            double part2 = 0;

            for (int i = 0; i < numClasses; i++) {
                part1 += sumOfSquares[i];
                part2 += sums[i];
            }

            part2 *= part2;
            part2 /= numInstances;
            ssTotal = part1 - part2;

            double ssAmoung;
            part1 = 0;
            part2 = 0;
            for (int i = 0; i < numClasses; i++) {
                part1 += (double) sumsSquared[i] / parentClassDist.get((double) i);//.data[i].size();
                part2 += sums[i];
            }
            ssAmoung = part1 - (part2 * part2) / numInstances;
            double ssWithin = ssTotal - ssAmoung;

            int dfAmoung = numClasses - 1;
            int dfWithin = numInstances - numClasses;

            double msAmoung = ssAmoung / dfAmoung;
            double msWithin = ssWithin / dfWithin;

            double f = msAmoung / msWithin;
            
            //Reset running sums
            multiplyer = 1;
            for (OrderLineObj currentObj : meanDistOrderLine) {
                double thisDist;
                int unassignedObjs = parentClassDist.get(currentObj.getClassVal()) - orderLineClassDist.get(currentObj.getClassVal());
                
                if(currentObj.getClassVal() == min.getClassVal()){
                    thisDist = minDistance;
                }else if(currentObj.getClassVal() == max.getClassVal()){
                    thisDist = maxDistance;
                }else{
                    thisDist = minDistance + (increment * multiplyer);
                    multiplyer++;        
                }
                sums[(int)currentObj.getClassVal()] -= thisDist * unassignedObjs;
                sumOfSquares[(int)currentObj.getClassVal()] -= thisDist * thisDist * unassignedObjs;
                sumsSquared[(int)currentObj.getClassVal()] = sums[(int)currentObj.getClassVal()] * sums[(int)currentObj.getClassVal()];
            }
            
            return Double.isNaN(f) ? 0.0 : f;
        }    
        
        @Override
        public boolean pruneCandidate(){
            if(orderLine.size() % parentClassDist.size() != 0){
                return false;
            }else{
                return super.pruneCandidate();
            }
        }  
                
    }   
