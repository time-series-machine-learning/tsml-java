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
package timeseriesweka.classifiers.distance_based.FastWWS.windowSearcher;

import java.util.ArrayList;
import java.util.Collections;

import timeseriesweka.classifiers.distance_based.FastWWS.items.LazyAssessNN;
import timeseriesweka.classifiers.distance_based.FastWWS.items.SequenceStatsCache;
import timeseriesweka.classifiers.distance_based.FastWWS.items.LazyAssessNN.RefineReturnType;
import timeseriesweka.classifiers.distance_based.FastWWS.sequences.SymbolicSequence;

/**
 * Code for the paper "Efficient search of the best warping window for Dynamic Time Warping" published in SDM18
 * 
 * Search for the best warping window using Fast Warping Window Search (FastWWS)
 * 
 * @author Chang Wei Tan, Francois Petitjean, Matthieu Herrmann, Germain Forestier, Geoff Webb
 *
 */
public class FastWWS extends WindowSearcher {
	// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Internal types
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
	/**
     * Potential nearest neighbour
     */
    private static class PotentialNN {
        /**
         * Status of the PotentialNN
         */
        public enum Status {
            NN,                         // This is the Nearest Neighbour
            BC,                         // Best Candidate so far
        }
        
        public int index;               // Index of the sequence in train[]
        public int r;                   // Window validity
        public double distance;         // Computed distance
        public Status status;           // Is that

        public PotentialNN() {
            this.index = Integer.MIN_VALUE;                 // Will be an invalid, negative, index.
            this.r = Integer.MAX_VALUE;						// Max: stands for "haven't found yet"
            this.distance = Double.POSITIVE_INFINITY;       // Infinity: stands for "not computed yet".
            this.status = Status.BC;                        // By default, we don't have any found NN.
        }

        /**
         * Setting the Potential NN for the query at a window
         * @param index: Index in training dataset
         * @param r: Window validity with query
         * @param distance: Distance to query
         * @param status: Status of the nearest neighbour
         */
        public void set(int index, int r, double distance, Status status) {
            this.index = index;
            this.r = r;
            this.distance = distance;
            this.status = status;
        }

        /** 
         * Check if this is a nearest neighbour for the query at a window
         * @return
         */
        public boolean isNN() {
            return this.status == Status.NN;
        }

        @Override
        public String toString() {
            return "" + this.index;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;

            PotentialNN that = (PotentialNN) o;

            return index == that.index;
        }
    }
    
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Fields
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    private static final long serialVersionUID = 1536192551485201554L;
    private PotentialNN[][] nns;                                        // Our main structure
    private boolean init;                                               // Have we initialize our structure?
    
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Constructor
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    public FastWWS() {
        super();
        forwardSearch = false;
        init = false;
    }
    
    public FastWWS(String name) {
        super();
        forwardSearch = false;
        init = false;
        datasetName = name;
    }

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Methods
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    public String doTime(long start){
        long duration = System.currentTimeMillis() - start;
        return "" + (duration / 1000) + " s " + (duration % 1000) + " ms";
    }

    public String doTime(long start, long now){
        long duration = now - start;
        return "" + (duration / 1000) + " s " + (duration % 1000) + " ms";
    }
    
    /**
     * Initializing our main structure
     * 
     * Data:
     * protected SymbolicSequence[] train; 									-- Array of sequences
     * protected HashMap<String, ArrayList<SymbolicSequence>> classedData;	-- Sequences by classes
     * protected HashMap<String, ArrayList<Integer>> classedDataIndices; 	-- Sequences index in train by classes
     * protected String[] classMap; 										-- Class per index
     * 
     * We are using SymbolicSequenceScoredClassed to retain what is the nearest neighbour:
     * this.public SymbolicSequence sequence; 				-- The sequence of interest, the nearest neighbour itself
     * this.public String classValue;						-- Class of the NN
     * this.public int index;                           	-- Index of the NN in the train
     * this.public public int smallestValidWindow;      	-- Smallest window that would give the same distance
     * this.public double score                         	-- Value of the distance
     * 
     * When computing DTW, we get a DTWResult, storing more data than just the distance:
     * this.public double distance;		-- the DTW distance
     * this.public int r;               -- the smallest window that would give the same path
     */
    protected void initTable() {
        if (train.length < 2) {
            System.err.println("Set is to small: " + train.length + " sequence. At least 2 sequences needed.");
        }

        System.out.println("Starting optimisation");

        //
        // --- STATS DECLARATIONS
        //
        // Timing and progress output
        long timeInit = System.currentTimeMillis();
        
        //
        // --- ALGORITHM DECLARATIONS & INITIALISATION
        //
        // Cache:
        SequenceStatsCache cache = new SequenceStatsCache(train, maxWindow);

        // We need a N*L storing area. We favorite an access per window size.
        // For each [Window Size][sequence], we store the nearest neighbour. See above.
        nns = new PotentialNN[maxWindow + 1][train.length];
        for (int win = 0; win < maxWindow + 1; ++win) {
            for (int len = 0; len < train.length; ++len) {
                nns[win][len] = new PotentialNN();
            }
        }

        // Vector of LazyUCR distance, propagating bound info "horizontally"
        LazyAssessNN[] lazyUCR = new LazyAssessNN[train.length];
        for (int i = 0; i < train.length; ++i) {
            lazyUCR[i] = new LazyAssessNN(cache);
        }
        // "Challengers" that compete with each other to be the NN of query
        ArrayList<LazyAssessNN> challengers = new ArrayList<LazyAssessNN>(train.length);

        System.out.println("Initialisation done ("+doTime(timeInit)+")");
        
        //
        // --- ALGORITHM
        //
        // Iteration for all TS, starting with the second one (first is a reference)
        for(int current=1; current < train.length; ++current){
            // --- --- Get the data --- ---
            SymbolicSequence sCurrent = train[current];

            // Clear off the previous challengers and add all the previous sequences
            challengers.clear();		
            for(int previous=0; previous < current; ++previous) {
                LazyAssessNN d = lazyUCR[previous];
                d.set(train[previous], previous, sCurrent, current);
                challengers.add(d);
            }

            // --- --- For each, decreasing (positive) windows --- ---
            for(int win = maxWindow; win > -1; --win){
                // --- Get the data
                PotentialNN currPNN = nns[win][current];

                if(currPNN.isNN()){
                    // --- --- WITH NN CASE --- ---
                    // We already have a NN for sure, but we still have to check if current is a new NN for previous
                    for(int previous = 0; previous < current; ++previous){
                        // --- Get the data
                        PotentialNN prevNN = nns[win][previous];

                        // --- Try to beat the previous best NN
                        double toBeat = prevNN.distance;
                        LazyAssessNN challenger = lazyUCR[previous];
                        RefineReturnType rrt = challenger.tryToBeat(toBeat, win);

                        // --- Check the result
                        if(rrt == RefineReturnType.New_best){
                            int r = challenger.getMinWindowValidityForFullDistance();
                            double d = challenger.getDistance(win);
                            prevNN.set(current, r, d, PotentialNN.Status.NN);
                        }
                    }
                } // END WITH NN CASE
                else {
                    // --- --- WITHOUT NN CASE --- ---
                    // We don't have a NN yet.
                    // Sort the challengers so we have a better chance to organize a good pruning.
                    Collections.sort(challengers);

                    for (LazyAssessNN challenger : challengers) {
                        // --- Get the data
                        int previous = challenger.indexQuery;
                        PotentialNN prevNN = nns[win][previous];

                        // --- First we want to beat the current best candidate:
                        double toBeat = currPNN.distance;
                        RefineReturnType rrt = challenger.tryToBeat(toBeat, win);

                        // --- Check the result
                        if (rrt == RefineReturnType.New_best) {
                            int r = challenger.getMinWindowValidityForFullDistance();
                            double d = challenger.getDistance(win);
                            currPNN.set(previous, r, d, PotentialNN.Status.BC);
                        }

                        // --- Now check for previous NN
                        // --- Try to beat the previous best NN
                        toBeat = prevNN.distance;
                        challenger = lazyUCR[previous];
                        rrt = challenger.tryToBeat(toBeat, win);

                        // --- Check the result
                        if (rrt == RefineReturnType.New_best) {
                            int r = challenger.getMinWindowValidityForFullDistance();
                            double d = challenger.getDistance(win);
                            prevNN.set(current, r, d, PotentialNN.Status.NN);
                        }
                    } // END for(AutoRefineDistance challenger: challengers)

                    // --- When we looked at every past sequences,
                    // the current best candidate is really the best one, so the NN.
                    // So assign the current NN to all the windows that are valid
                    int r = currPNN.r;
                    double d = currPNN.distance;
                    int index = currPNN.index;
                    for (int w = win; w >= r; --w) {
                        nns[w][current].set(index, r, d, PotentialNN.Status.NN);
                    }
                } // END WITHOUT NN CASE
            } // END for(int win=maxWindow; win>-1; --win)
        } // END for(int current=1; current < train.length; ++current)

        System.out.println("done! (" + doTime (timeInit) + ")");
        this.init = true;        
    } // END initTable()

    @Override
    protected double evalSolution(int warpingWindow) {
        // Will only be called once
        if (!init) {
            initTable();
        }

        // Error counter:
        int nErrors = 0;

        for (int i = 0; i < train.length; i++) {
            if (!classMap[nns[warpingWindow][i].index].equals(classMap[i])) {
                nErrors++;
            }
        }

        return 1.0 * nErrors / train.length;
    }
     
    /** 
     * Get the best warping window found
     */
    @Override
	public int getBestWin() {
    	return bestWarpingWindow;
    }
    
    /** 
     * Get the LOOCV accuracy for the best warping window
     */
    @Override
	public double getBestScore() {
    	return bestScore;
    }
    
    /** 
     * Set the result directory
     */
    @Override
	public void setResDir(String path) {
    	resDir = path;
    }
    
    /** 
     * Set type of classifier
     */
    @Override
	public void setType(String t) {
    	type = t;
    }
}
