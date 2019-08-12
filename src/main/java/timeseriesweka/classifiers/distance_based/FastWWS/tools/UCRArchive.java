/*******************************************************************************
 * Copyright (C) 2017 Chang Wei Tan
 * 
 * This file is part of FastWWSearch.
 * 
 * FastWWSearch is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 * 
 * FastWWSearch is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with FastWWSearch.  If not, see <http://www.gnu.org/licenses/>.
 ******************************************************************************/
package timeseriesweka.classifiers.distance_based.FastWWS.tools;

/** 
 * Stores dataset names for the Standard UCR Archive 
 * 
 * Yanping Chen, Eamonn Keogh, Bing Hu, Nurjahan Begum, Anthony Bagnall, Abdullah Mueen and Gustavo Batista (2015). 
 * The UCR Time Series Classification Archive. 
 * URL www.cs.ucr.edu/~eamonn/time_series_data/
 * 
 * @author Chang Wei Tan
 *
 */
public class UCRArchive {
	/**
	 * Sorted in increasing DTW computations per test series
	 */
	public static String[] sortedDataset = new String[]{"SonyAIBORobotSurface","ItalyPowerDemand",
			"MoteStrain","SonyAIBORobotSurfaceII","TwoLeadECG","ECGFiveDays","CBF",
			"DiatomSizeReduction","Gun_Point","Coffee","FaceFour","ArrowHead","ECG200",
			"Symbols","ShapeletSim","BeetleFly","BirdChicken","ToeSegmentation1",
			"DistalPhalanxOutlineAgeGroup","DistalPhalanxTW","MiddlePhalanxOutlineAgeGroup",
			"MiddlePhalanxTW","ToeSegmentation2","Wine","Beef","Plane","ProximalPhalanxTW",
			"OliveOil","synthetic_control","DistalPhalanxOutlineCorrect","Lighting7",
			"MiddlePhalanxOutlineCorrect","FacesUCR","Meat","Trace","ProximalPhalanxOutlineAgeGroup",
			"Herring","Car","MedicalImages","Lighting2","Ham","ProximalPhalanxOutlineCorrect",
			"InsectWingbeatSound","MALLAT","SwedishLeaf","CinC_ECG_torso","Adiac","Worms","WormsTwoClass",
			"ECG5000","Earthquakes","WordsSynonyms","FaceAll","ChlorineConcentration","FISH","OSULeaf",
			"Strawberry","Cricket_X","Cricket_Y","Cricket_Z","50words","yoga","Two_Patterns",
			"PhalangesOutlinesCorrect","wafer","Haptics","Computers","InlineSkate","Phoneme",
			"LargeKitchenAppliances","RefrigerationDevices","ScreenType","SmallKitchenAppliances",
			"uWaveGestureLibrary_X","uWaveGestureLibrary_Y","uWaveGestureLibrary_Z","ShapesAll",
			"FordB","FordA","UWaveGestureLibraryAll","ElectricDevices","HandOutlines",
			"StarLightCurves","NonInvasiveFatalECG_Thorax1","NonInvasiveFatalECG_Thorax2"};
	
	/**
	 * Datasets that are small and fast to classify
	 */
	public static String[] smallDataset = new String[]{"SonyAIBORobotSurface","ItalyPowerDemand",
			"MoteStrain","SonyAIBORobotSurfaceII","TwoLeadECG","ECGFiveDays","CBF",
			"DiatomSizeReduction","Gun_Point","Coffee","FaceFour","ArrowHead","ECG200",
			"Symbols","ShapeletSim","BeetleFly","BirdChicken","ToeSegmentation1",
			"DistalPhalanxOutlineAgeGroup","DistalPhalanxTW","MiddlePhalanxOutlineAgeGroup",
			"MiddlePhalanxTW","ToeSegmentation2","Wine","Beef","Plane","ProximalPhalanxTW",
			"OliveOil","synthetic_control","DistalPhalanxOutlineCorrect","Lighting7",
			"MiddlePhalanxOutlineCorrect","FacesUCR","Meat","Trace","ProximalPhalanxOutlineAgeGroup",
			"Herring","Car","MedicalImages","Lighting2","Ham","ProximalPhalanxOutlineCorrect",
			"InsectWingbeatSound","MALLAT","SwedishLeaf","CinC_ECG_torso","Adiac","Worms","WormsTwoClass",
			"ECG5000","Earthquakes","WordsSynonyms","FaceAll","ChlorineConcentration","FISH","OSULeaf",
			"Strawberry","Cricket_X","Cricket_Y","Cricket_Z","50words","yoga"};
	
	/** 
	 * New datasets used in 
	 * Time-Series Classification with COTE: The Collective of Transformation-Based Ensembles (COTE) and 
	 * Time series classification with ensembles of elastic distance measures (EE)
	 * 
	 * Refer to http://www.timeseriesclassification.com/dataset.php
	 */
	public static String[] newTSCProblems = new String[] {"ElectricDeviceOn","EpilepsyX",
			"EthanolLevel","HeartbeatBIDMC","NonInvasiveFetalECGThorax1","NonInvasiveFetalECGThorax2"};
	
	/**
	 * Size of all dataset
	 * @return
	 */
	public static int totalDatasets(){
		return newTSCProblems.length + sortedDataset.length;
	}
	
	public static String total(){
		return newTSCProblems.length + sortedDataset.length + " in total.";
	}
}
