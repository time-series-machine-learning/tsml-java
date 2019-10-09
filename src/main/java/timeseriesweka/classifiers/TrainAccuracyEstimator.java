///*
// *   This program is free software: you can redistribute it and/or modify
// *   it under the terms of the GNU General Public License as published by
// *   the Free Software Foundation, either version 3 of the License, or
// *   (at your option) any later version.
// *
// *   This program is distributed in the hope that it will be useful,
// *   but WITHOUT ANY WARRANTY; without even the implied warranty of
// *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// *   GNU General Public License for more details.
// *
// *   You should have received a copy of the GNU General Public License
// *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
// */
//package timeseriesweka.classifiers;
//
//import evaluation.storage.ClassifierResults;
//
///**
// * Classifiers implementing this interface are able to estimate their own performance 
// * (possibly with some bias) on the train data in some way as part of their buildClassifier process, 
// * and avoid a full nested-cross validation process.
// * 
// * The estimation process may be entirely encapsulated in the build process (e.g. a tuned 
// * classifier returning the train estimate of the best parameter set, acting as the train 
// * estimate of the full classifier: note the bias), or may be done as an
// * additional step beyond the normal build process but far more efficiently than a
// * nested cv (e.g. a 1NN classifier could perform an efficient internal loocv)  
// * 
// * Implementation of this interface indicates the ABILITY to estimate train performance, 
// * to turn this behaviour on, setFindingTrainPerformanceEstimate(true) should be called. 
// * By default, the estimation behaviour is off
// * 
// * This way, if for whatever reason a nested estimation process is explicitly wanted 
// * (e.g. for completely bias-free estimates), that can also be achieved
// * 
// * @author James Large (james.large@uea.ac.uk), ajb
// */
//public interface TrainAccuracyEstimator{
//
//    
//    /**
//     * For almost all classifiers, which extend both this interface and AbstractClassifierWithTrainingInfo,
//     * this method simply mirrors the AbstractClassifierWithTrainingInfo default implementation 
//     * and calls that, without needing to implement the method in your classifier
//     * 
//     * Gets the train results for this classifier, which will be empty (but not-null) 
//     * until buildClassifier has been called.
//     * 
//     * If the classifier was set-up to estimate  it's own train accuracy, these
//     * will be populated with full prediction information, ready to be written as a 
//     * trainFoldX file for example
//     * 
//     * Otherwise, the object will at minimum contain build time and parameter information
//     */
//    public ClassifierResults getTrainResults();
//    
//    /**
//     * For almost all classifiers, which extend both this interface and AbstractClassifierWithTrainingInfo,
//     * this method simply mirrors the AbstractClassifierWithTrainingInfo default implementation 
//     * and calls that, without needing to implement the method in your classifier
//     * 
//     * Determines whether this classifier should generates a performance estimate on the 
//     * train data internally during the buildclassifier process. 
//     * 
//     * Default behaviour is not to find them. In this case, the only information in trainResults
//     * relates to the time taken to build the classifier
//     */
//    public void setEstimatingPerformanceOnTrain(boolean b);
//    
//    /** 
//     * For almost all classifiers, which extend both this interface and AbstractClassifierWithTrainingInfo,
//     * this method simply mirrors the AbstractClassifierWithTrainingInfo default implementation 
//     * and calls that, without needing to implement the method in your classifier
//     * 
//     * Determines whether this classifier should generates a performance estimate on the 
//     * train data internally during the buildclassifier process. 
//     * 
//     * Default behaviour is not to find them. In this case, the only information in trainResults
//     * relates to the time taken to build the classifier
//     */
//    public boolean getEstimatingPerformanceOnTrain();
//
//}
