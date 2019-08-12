/*******************************************************************************
 * Copyright (C) 2017 Chang Wei Tan, Francois Petitjean, Matthieu Herrmann, Germain Forestier, Geoff Webb
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
package timeseriesweka.classifiers.distance_based.FastWWS.items;

import java.io.File;

import timeseriesweka.classifiers.distance_based.FastWWS.tools.UCR2CSV;
import weka.core.Instances;
import weka.core.converters.CSVLoader;

/**
 * Code for the paper "Efficient search of the best warping window for Dynamic Time Warping" published in SDM18
 * 
 * @author Chang Wei Tan, Francois Petitjean, Matthieu Herrmann, Germain Forestier, Geoff Webb
 *
 */
public class ExperimentsLauncher {
	// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Fields
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
	public static String username = System.getProperty("user.name");
	long startTime;
	long endTime;
	long duration;
	
	// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Constructor
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
	public ExperimentsLauncher() {

	}

	// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Method
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
	/**
	 * Read train and test set from the given path 
	 * @param path
	 * @param name
	 * @return
	 */
	public static Instances[] readTrainAndTest(String path, String name) {
		File trainFile = new File(path + name + "/" + name + "_TRAIN");
		if (!new File(trainFile.getAbsolutePath() + ".csv").exists()) {
			UCR2CSV.run(trainFile, new File(trainFile.getAbsolutePath() + ".csv"));
		}
		trainFile = new File(trainFile.getAbsolutePath() + ".csv");
		
		File testFile = new File(path + name + "/" + name + "_TEST");
		if (!new File(testFile.getAbsolutePath() + ".csv").exists()) {
			UCR2CSV.run(testFile, new File(testFile.getAbsolutePath() + ".csv"));
		}
		testFile = new File(testFile.getAbsolutePath() + ".csv");

		CSVLoader loader = new CSVLoader();
		Instances trainDataset = null;
		Instances testDataset = null;

		try {
			loader.setFile(trainFile);
			loader.setNominalAttributes("first");
			trainDataset = loader.getDataSet();
			trainDataset.setClassIndex(0);

			loader.setFile(testFile);
			loader.setNominalAttributes("first");
			testDataset = loader.getDataSet();
			testDataset.setClassIndex(0);
		} catch (Exception e) {
			e.printStackTrace();
		}
		return new Instances[] { trainDataset, testDataset };
	}
	
	/**
	 * Read both train and test set from a given path and merge them both
	 * @param path
	 * @param name
	 * @return
	 */
	public static Instances readAllInOne(String path, String name) {
		File trainFile = new File(path + name + "/" + name + "_TRAIN");
		if (!new File(trainFile.getAbsolutePath() + ".csv").exists()) {
			UCR2CSV.run(trainFile, new File(trainFile.getAbsolutePath() + ".csv"));
		}
		trainFile = new File(trainFile.getAbsolutePath() + ".csv");
		
		File testFile = new File(path + name + "/" + name + "_TEST");
		if (!new File(testFile.getAbsolutePath() + ".csv").exists()) {
			UCR2CSV.run(testFile, new File(testFile.getAbsolutePath() + ".csv"));
		}
		testFile = new File(testFile.getAbsolutePath() + ".csv");

		CSVLoader loader = new CSVLoader();
		Instances trainDataset = null;
		Instances testDataset = null;

		try {
			loader.setFile(trainFile);
			loader.setNominalAttributes("first");
			trainDataset = loader.getDataSet();
			trainDataset.setClassIndex(0);
			
			loader.setFile(testFile);
			loader.setNominalAttributes("first");
			testDataset = loader.getDataSet();
			testDataset.setClassIndex(0);
			
			for (int i = 0; i < testDataset.numInstances(); i++) {
				trainDataset.add(testDataset.instance(i));
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		return trainDataset;
	}	
}
