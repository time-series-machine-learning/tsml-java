/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package timeseriesweka.filters.shapelet_transforms.quality_measures;

import java.util.ArrayList;
import java.util.Collections;
import utilities.class_distributions.ClassDistribution;
import timeseriesweka.filters.shapelet_transforms.OrderLineObj;

/**
 *
 * @author raj09hxu
 */
/**
     * A class for calculating the Kruskal Wallis statistic bound of a shapelet, according to 
     * the set of distances from the shapelet to a dataset.
     */
    public class KruskalWallisBound extends ShapeletQualityBound{
        
        /**
         * Constructor to construct KruskalWallisBound
         * @param classDist class distribution of the data currently being processed
         * @param percentage percentage of data required to be processed before
         *                   bounding mechanism is used.
         */
        protected KruskalWallisBound(ClassDistribution classDist, int percentage){
            initParentFields(classDist, percentage);
        }
               
        @Override
        public void updateOrderLine(OrderLineObj orderLineObj){
            super.updateOrderLine(orderLineObj);
            numInstances--;
        }
               
        /**
         * Method to calculate the quality bound for the current orderline
         * @return Kruskal Wallis statistic bound
         */
        @Override
        protected double calculateBestQuality() {
             
            //1) Find sums of ranks for the observed orderline objects
            int numClasses = parentClassDist.size();
            int[] classRankCounts = new int[numClasses];
            double minimumRank = -1.0;
            double maximumRank = -1.0;
            double lastDistance = orderLine.get(0).getDistance();
            double thisDistance;
            double classVal = orderLine.get(0).getClassVal();
            classRankCounts[(int)classVal]+=1;

            int duplicateCount = 0;

            for(int i=1; i< orderLine.size(); i++){
                thisDistance = orderLine.get(i).getDistance();
                if(duplicateCount == 0 && thisDistance!=lastDistance){ // standard entry
                    classRankCounts[(int)orderLine.get(i).getClassVal()]+=i+1;
                    
                    //Set min/max ranks
                    if(thisDistance > 0.0 && minimumRank == -1.0){
                        minimumRank = i+1;
                    }
                    maximumRank = i+1;
                }else if(duplicateCount > 0 && thisDistance!=lastDistance){ // non-duplicate following duplicates
                    // set ranks for dupicates

                    double minRank = i-duplicateCount;
                    double maxRank = i;
                    double avgRank = (minRank+maxRank)/2;

                    for(int j = i-duplicateCount-1; j < i; j++){
                        classRankCounts[(int)orderLine.get(j).getClassVal()]+=avgRank;
                    }


                    duplicateCount = 0;
                    // then set this rank
                    classRankCounts[(int)orderLine.get(i).getClassVal()]+=i+1;
                   
                    //Set min/max ranks
                    if(thisDistance > 0.0 && minimumRank == -1.0){
                        minimumRank = i+1;
                    }
                    maximumRank = i+1;
                } else{// thisDistance==lastDistance
                    if(i == orderLine.size() - 1){ // last one so must do the avg ranks here (basically copied from above, BUT includes this element too now)

                        double minRank = i-duplicateCount;
                        double maxRank = i+1;
                        double avgRank = (minRank+maxRank)/2;

                        for(int j = i-duplicateCount-1; j <= i; j++){
                            classRankCounts[(int)orderLine.get(j).getClassVal()]+=avgRank;
                        }
                        
                        //Set min/max ranks
                        if(thisDistance > 0.0 && minimumRank == -1.0){
                            minimumRank = avgRank;
                        }
                        maximumRank = avgRank;  
                    }
                    duplicateCount++;
                }
                lastDistance = thisDistance;
            }

            // 2) Compute mean rank for the obsereved objects 
            ArrayList<OrderLineObj> meanRankOrderLine = new ArrayList<>();
            for(int i = 0; i < numClasses; i++){
                meanRankOrderLine.add(new OrderLineObj((double)classRankCounts[i]/orderLineClassDist.get((double)i), (double)i));
            }
            Collections.sort(meanRankOrderLine);
            
            //Find approximate minimum orderline objects
            OrderLineObj min = new OrderLineObj(-1.0, 0.0);
            for (OrderLineObj meanRankOrderLine1 : meanRankOrderLine) {
                classVal = meanRankOrderLine1.getClassVal();
                int unassignedObjs = parentClassDist.get(classVal) - orderLineClassDist.get(classVal);
                double observed = classRankCounts[(int)classVal];
                double predicted = minimumRank * unassignedObjs;
                double approximateRank = (observed + predicted) / parentClassDist.get(classVal);
                if(min.getDistance() == -1.0 || approximateRank < min.getDistance()){
                    min.setDistance(approximateRank);
                    min.setClassVal(classVal);
                }
            }
            
            //Find approximate maximum orderline objects
            OrderLineObj max = new OrderLineObj(-1.0, 0.0);
            for (OrderLineObj meanRankOrderLine1 : meanRankOrderLine) {
                classVal = meanRankOrderLine1.getClassVal();
                int unassignedObjs = parentClassDist.get(classVal) - orderLineClassDist.get(classVal);
                double observed = classRankCounts[(int)classVal];
                double predicted = maximumRank * unassignedObjs;
                double approximateRank = (observed + predicted) / parentClassDist.get(classVal); 
                if(classVal != min.getClassVal() && (max.getDistance() == -1.0 || approximateRank > max.getDistance())){
                    max.setDistance(approximateRank);
                    max.setClassVal(classVal);
                }
            }
            
            //3) overall mean rank
            double overallMeanRank = (1.0+ orderLine.size() + numInstances)/2;
    
            //4) Interpolate mean ranks
            double increment = (max.getDistance() - min.getDistance()) / (numClasses-1);
            int multiplyer = 1;
            for (OrderLineObj currentObj : meanRankOrderLine) {
                int unassignedObjs = parentClassDist.get(currentObj.getClassVal()) - orderLineClassDist.get(currentObj.getClassVal());
                
                if(currentObj.getClassVal() == min.getClassVal()){
                    currentObj.setDistance(min.getDistance());
                }else if(currentObj.getClassVal() == max.getClassVal()){
                    currentObj.setDistance(max.getDistance());
                }else{
                    classVal = currentObj.getClassVal();
                    double observed = classRankCounts[(int)classVal];
                    double predicted = (minimumRank + (increment * multiplyer)) * unassignedObjs;
                    double approximateRank = (observed + predicted) / parentClassDist.get(classVal); 
                    currentObj.setDistance(approximateRank);
                    multiplyer++;        
                }
            }
            
            //5) sum of squared deviations from the overall mean rank
            double s = 0;
            for(int i = 0; i < numClasses; i++){
                s+= parentClassDist.get((double)i)*(meanRankOrderLine.get(i).getDistance() -overallMeanRank)*(meanRankOrderLine.get(i).getDistance() -overallMeanRank);
            }

            //6) weight s with the scale factor
            int totalInstances = orderLine.size() + numInstances;
            double h = 12.0/(totalInstances*(totalInstances+1))*s;

            return h;
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
