/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package timeseriesweka.filters.shapelet_transforms.quality_measures;
import java.util.Map;
import java.util.TreeMap;
import utilities.class_distributions.ClassDistribution;
import utilities.class_distributions.TreeSetClassDistribution;
/**
 *
 * @author raj09hxu
 */
public class InformationGainBound extends ShapeletQualityBound{
        private double parentEntropy;
        boolean isExact;
        
        /**
         * Constructor to construct InformationGainBound
         * @param classDist class distribution of the data currently being processed
         * @param percentage percentage of data required to be processed before
         *                   bounding mechanism is used.
         *
         * @param isExact
         * */
        protected InformationGainBound(ClassDistribution classDist, int percentage, boolean isExact){
            initParentFields(classDist, percentage);
            this.isExact = isExact;
            parentEntropy = InformationGain.entropy(parentClassDist);
        }
        protected InformationGainBound(ClassDistribution classDist, int percentage){
            this(classDist,percentage,false);
        }
           
        /**
         * Method to calculate the quality bound for the current orderline
         * @return information gain bound
         */
        @Override
        protected double calculateBestQuality(){
            Map<Double, Boolean> perms = new TreeMap<>();
            double bsfGain = -1;
                        
            //Cycle through all permutations
            if(isExact){
                //Initialise perms
                for(Double key : orderLineClassDist.keySet()){
                    perms.put(key, Boolean.TRUE);
                }
            
                for(int totalCycles = perms.keySet().size(); totalCycles > 1; totalCycles--){
                    for(int cycle = 0; cycle < totalCycles; cycle++){
                        int start = 0, count = 0;
                        for(Double key : perms.keySet()){
                            Boolean val = Boolean.TRUE;
                            if(cycle == start){
                                val = Boolean.FALSE;
                                int size = perms.keySet().size();
                                if(totalCycles <  size && count < (size - totalCycles)){
                                    count++;
                                    start--;
                                }
                            }
                            perms.put(key, val);
                            start++;
                        }
                        //Check quality of current permutation
                        double currentGain = computeIG(perms);

                        if(currentGain > bsfGain){
                            bsfGain = currentGain;
                        }

                        if(bsfGain > bsfQuality){
                            break;
                        }

                    }
                }
            }else{
                double currentGain = computeIG(null);
                    
                if(currentGain > bsfGain){
                    bsfGain = currentGain;
                }
            }
        
            return bsfGain;
        }
        
        private double computeIG(Map<Double, Boolean> perm){
            //Initialise class counts
            TreeSetClassDistribution lessClasses = new TreeSetClassDistribution();
            TreeSetClassDistribution greaterClasses = new TreeSetClassDistribution();
            TreeMap<Double, Boolean> isShifted = new TreeMap<>();
            
            int countOfAllClasses = 0;
            int countOfLessClasses = 0;
            int countOfGreaterClasses = 0;
            
            for(double j : parentClassDist.keySet()){
                int lessVal =0;
                int greaterVal = parentClassDist.get(j);
                
                if(perm != null){
                    if(perm.get(j) != null && perm.get(j)){
                        lessVal = parentClassDist.get(j) - orderLineClassDist.get(j);
                        greaterVal = orderLineClassDist.get(j);
                    }
                    countOfLessClasses += lessClasses.get(j);
                
               //Assign everything to the right for fast bound
                }else{
                    isShifted.put(j, Boolean.FALSE);
                }
                
                lessClasses.put(j, lessVal);
                greaterClasses.put(j, greaterVal);
                countOfGreaterClasses += greaterClasses.get(j);
                
                
                countOfAllClasses += parentClassDist.get(j);
            }
           

            double bsfGain = -1;
            double lastDist = -1;
            double thisDist;
            double thisClassVal;
            int oldCount;

            for(int i = 0; i < orderLine.size()-1; i++){ 
                thisDist = orderLine.get(i).getDistance();
                thisClassVal = orderLine.get(i).getClassVal();

                 //move the threshold along one (effectively by adding this dist to lessClasses
                oldCount = lessClasses.get(thisClassVal)+1;
                lessClasses.put(thisClassVal,oldCount);                
                oldCount = greaterClasses.get(thisClassVal)-1;
                greaterClasses.put(thisClassVal,oldCount);
                
                // adjust counts - maybe makes more sense if these are called counts, rather than sums!
                countOfLessClasses++;
                countOfGreaterClasses--;

                //For fast bound dynamically shift the unassigned objects when majority side changes
                if(!isExact){
                    //Check if shift has not already happened
                    if(!isShifted.get(thisClassVal)){
                        int greaterCount = greaterClasses.get(thisClassVal) - (parentClassDist.get(thisClassVal) - orderLineClassDist.get(thisClassVal));
                        int lessCount = lessClasses.get(thisClassVal);
                        
                        //Check if shift has happened
                        if(lessCount - greaterCount > 0){
                            greaterClasses.put(thisClassVal, greaterClasses.get(thisClassVal) - (parentClassDist.get(thisClassVal) - orderLineClassDist.get(thisClassVal)));
                            countOfGreaterClasses -= parentClassDist.get(thisClassVal) - orderLineClassDist.get(thisClassVal);
                            lessClasses.put(thisClassVal, lessClasses.get(thisClassVal) + (parentClassDist.get(thisClassVal) - orderLineClassDist.get(thisClassVal)));
                            countOfLessClasses += parentClassDist.get(thisClassVal) - orderLineClassDist.get(thisClassVal);
                            isShifted.put(thisClassVal, Boolean.TRUE);
                        }
                    }
                }
                
                // check to see if the threshold has moved (ie if thisDist isn't the same as lastDist)
                // important, else gain calculations will be made 'in the middle' of a threshold, resulting in different info gain for
                // the split point, that won't actually be valid as it is 'on' a distances, rather than 'between' them/
                if(thisDist != lastDist){

                    // calculate the info gain below the threshold
                    double lessFrac =(double) countOfLessClasses / countOfAllClasses;
                    double entropyLess = InformationGain.entropy(lessClasses);

                    // calculate the info gain above the threshold
                    double greaterFrac =(double) countOfGreaterClasses / countOfAllClasses;
                    double entropyGreater = InformationGain.entropy(greaterClasses);

                    double gain = parentEntropy - lessFrac * entropyLess - greaterFrac * entropyGreater;
                    if(gain > bsfGain){
                        bsfGain = gain;
                    }
                }
                
                lastDist = thisDist;
            }
            
            return bsfGain;
        }
        
        
        @Override
        public boolean pruneCandidate(){
            //Check if we at least have observed an object from each class
            if(orderLine.size() % parentClassDist.size() != 0){
                return false;
            }else{
                return super.pruneCandidate();
            }
        }          
    }
