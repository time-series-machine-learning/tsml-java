package timeseriesweka.classifiers.distance_based.FastWWS.experiments;

import java.io.File;
import java.io.FileFilter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;

import timeseriesweka.classifiers.distance_based.FastWWS.items.ExperimentsLauncher;
import timeseriesweka.classifiers.distance_based.FastWWS.tools.Sampling;
import timeseriesweka.classifiers.distance_based.FastWWS.tools.UCRArchive;
import timeseriesweka.classifiers.distance_based.FastWWS.windowSearcher.FastWWS;
import timeseriesweka.classifiers.distance_based.FastWWS.windowSearcher.FastWWSPrunedDTW;
import weka.classifiers.Evaluation;
import weka.core.Instances;

public class UCR_FastWWSPrunedDTW {
	private static String osName, datasetName, username, projectPath, datasetPath, resDir, sampleType, method;
	private static int bestWarpingWindow;
	private static double bestScore;
	private static int nbRuns = 1;
	private static boolean firstRun = false;
	
	public static void main(String[] args) throws Exception {
		// Initialise
		sampleType = "Single";				// Doing just 1 dataset, can be Sorted, Small, New or All
		datasetName = "ElectricDevices";	// Name of dataset to be tested
		method = "FastWWSearch-PrunedDTW";	// Method type in finding the best window
		
		// Get project and dataset path
		osName = System.getProperty("os.name");		
    	username = System.getProperty("user.name");
    	if (osName.contains("Window")) {
    		projectPath = "C:/Users/" + username + "/workspace/SDM18/";
    		if (sampleType.equals("New")) 
    			datasetPath = "C:/Users/" + username + "/workspace/Dataset/TSC_Problems/";
    		else
    			datasetPath = "C:/Users/" + username + "/workspace/Dataset/UCR_Time_Series_Archive/";
    	} else {
    		projectPath = "/home/" + username + "/workspace/SDM18/";
    		if (sampleType.equals("New")) 
    			datasetPath = "/home/" + username + "/workspace/Dataset/TSC_Problems/";
    		else
    			datasetPath = "/home/" + username + "/workspace/Dataset/UCR_Time_Series_Archive/";
    	}
					
		// Get arguments 
		if (args.length >= 1) projectPath = args[0];
		if (args.length >= 2) datasetPath = args[1];
		if (args.length >= 3) sampleType = args[2];
		if (sampleType.equals("Single") && args.length >= 4) {
			datasetName = args[3];
			if (args.length >= 5) nbRuns = Integer.parseInt(args[4]);
		} else if (args.length >= 4) {
			nbRuns = Integer.parseInt(args[3]);
		} 
		
		if (sampleType.equals("Single")) 
			System.out.println("Find best warping window with " + method + " on " + datasetName + " dataset -- " + nbRuns + " runs");
		else
			System.out.println("Find best warping window with " + method + " on " + sampleType + " dataset -- " + nbRuns + " runs");
				
		
		switch(sampleType) {
		case "Sorted":
			System.out.println("Find best warping window with Sorted UCR datasets");
			for (int j = 0; j < UCRArchive.sortedDataset.length; j++) {
				datasetName = UCRArchive.sortedDataset[j];
				singleProblem(datasetName);
			}
			break;
		case "Small":
			System.out.println("Find best warping window with Small UCR datasets");
			for (int j = 0; j < UCRArchive.smallDataset.length; j++) {
				datasetName = UCRArchive.smallDataset[j];
				singleProblem(datasetName);
			}
			break;
		case "New":
			System.out.println("Find best warping window with New TSC datasets");
			for (int j = 0; j < UCRArchive.newTSCProblems.length; j++) {
				datasetName = UCRArchive.newTSCProblems[j];
				singleProblem(datasetName);
			}
			break;
		case "All":
			System.out.println("Find best warping window with " + datasetName + " dataset");
			File rep = new File(datasetPath);
	        File[] listData = rep.listFiles(new FileFilter() {
	            @Override
	            public boolean accept(File pathname) {
	                return pathname.isDirectory();
	            }
	        });
	        Arrays.sort(listData);

	        for (File dataRep : listData) {	                        
	            datasetName = dataRep.getName();
	            singleProblem(datasetName);
	        }
	        break;
		case "Single":
			System.out.println("Find best warping window with " + datasetName + " dataset");
			singleProblem(datasetName);
			break;
		}		
        
    }// End main
	
	private static void singleProblem (String datasetName) throws Exception {
		resDir = projectPath + "outputs/Incorporate_PrunedDTW/" + datasetName + "/";
		
		File dir = new File(resDir);
        if (!dir.exists())
        	dir.mkdirs();
        
		double[][] searchTimes = new double[2][nbRuns];
        double speedUp = 0;
        double avgFastWWSTime = 0, avgFastWWSSDM16Time = 0;
        
        System.out.println("Processing: " + datasetName);
        Instances[] data = ExperimentsLauncher.readTrainAndTest(datasetPath, datasetName);

        Instances train = data[0];
        Instances test = data[1];
        
        // somehow need to have a dummy run otherwise, the first run will have significantly longer time for smaller datasets
        if (!firstRun) {
        	FastWWS tempClassifier = new FastWWS(datasetName);
        	tempClassifier.setResDir(resDir);
        	tempClassifier.setType(method);
        	tempClassifier.buildClassifier(train);
        	firstRun = true;
        }
    	
        for (int i = 0; i < nbRuns; i++) {
        	train = Sampling.random(train);        	
        	
        	method = "FastWWSearch";
        	System.out.println("Run " + i + ", Launching " + method);
        	FastWWS fastwwsClassifier = new FastWWS(datasetName);
        	fastwwsClassifier.setResDir(resDir);
        	fastwwsClassifier.setType(method);
        	long start = System.nanoTime();
        	fastwwsClassifier.buildClassifier(train);
        	long stop = System.nanoTime();
        	double searchTime = (double) ((stop - start)/1e9);
        	System.out.println(searchTime + " s");
        	searchTimes[0][i] = searchTime;
        	avgFastWWSTime += searchTime;
        	
        	bestWarpingWindow = fastwwsClassifier.getBestWin();
        	bestScore = fastwwsClassifier.getBestScore();

        	Evaluation eval = new Evaluation(train);
        	eval.evaluateModel(fastwwsClassifier, test);
        	System.out.println(eval.errorRate());

        	saveSearchTime(searchTime, eval.errorRate());
        	        	
        	method = "FastWWSearch-PrunedDTW";
        	System.out.println("Run " + i + ", Launching FastWWS with SDM16");
        	FastWWSPrunedDTW fastwwsSDM16Classifier = new FastWWSPrunedDTW(datasetName);
        	fastwwsSDM16Classifier.setResDir(resDir);
        	fastwwsSDM16Classifier.setType(method);
        	start = System.nanoTime();
        	fastwwsSDM16Classifier.buildClassifier(train);
        	stop = System.nanoTime();
        	searchTime = (double) ((stop - start)/1e9);
        	System.out.println(searchTime + " s");
        	searchTimes[1][i] = searchTime;
        	avgFastWWSSDM16Time += searchTime;
        	
        	bestWarpingWindow = fastwwsSDM16Classifier.getBestWin();
        	bestScore = fastwwsSDM16Classifier.getBestScore();

        	eval = new Evaluation(train);
        	eval.evaluateModel(fastwwsSDM16Classifier, test);
        	System.out.println(eval.errorRate());

        	saveSearchTime(searchTime, eval.errorRate());
        	
        	speedUp += searchTimes[0][i]/searchTimes[1][i];
        	
        }
        
        System.out.println("Average FastWWS Time: " + avgFastWWSTime/nbRuns + "s");
        System.out.println("Average FastWWS with SDM16 Time: " + avgFastWWSSDM16Time/nbRuns + "s");
        
        System.out.println("Average Speedup: " + speedUp/nbRuns);
	}
	
	private static void saveSearchTime(double searchTime, double error) {
		String fileName = resDir + datasetName + "_result_" + method + ".csv";
		FileWriter out;
		boolean append = false;
		File file = new File(fileName);
		if (file.exists()) 
			append = true;
		try {
			out = new FileWriter(fileName, append);
			if (!append)
				out.append("SearchTime(s),BestWin,BestScore,TestError\n");
			out.append(searchTime + "," + bestWarpingWindow + "," + bestScore + "," + error + "\n");
			out.flush();
	        out.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
