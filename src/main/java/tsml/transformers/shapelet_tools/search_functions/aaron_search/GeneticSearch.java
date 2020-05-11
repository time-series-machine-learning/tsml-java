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
package tsml.transformers.shapelet_tools.search_functions.aaron_search;

import java.util.ArrayList;
import java.util.List;

import tsml.transformers.shapelet_tools.search_functions.ShapeletSearch;
import tsml.transformers.shapelet_tools.search_functions.ShapeletSearchOptions;
import utilities.generic_storage.Pair;
import weka.core.Instance;
import weka.core.Instances;
import tsml.transformers.shapelet_tools.Shapelet;

/**
 *
 * @author raj09hxu
 */
@Deprecated
public class GeneticSearch extends ImprovedRandomSearch {

    int initialPopulationSize = 50;
    private int initialNumShapeletsPerSeries;
    private int evaluated;
    
    public GeneticSearch(ShapeletSearchOptions sop){
        super(sop);
    }
    
    @Override
    public void init(Instances input){
        super.init(input);
       
        initialNumShapeletsPerSeries = (int) (numShapeletsPerSeries / inputData.numInstances());
    }

    @Override
    public ArrayList<Shapelet> searchForShapeletsInSeries(Instance timeSeries, ShapeletSearch.ProcessCandidate checkCandidate){
       evaluated = 0;
        
       double[] series = timeSeries.toDoubleArray();
        
       List<Shapelet> population = new ArrayList<>();
       
        //generate the random shapelets we're going to visit.
        for(int i=0; i<initialPopulationSize; i++){
            //randomly generate values.

            Pair<Integer, Integer> pair = createRandomShapelet(series);
            Shapelet shape = checkCandidate.process(timeSeries, pair.var2, pair.var1);
            evaluated++;
            if(shape != null)
                population.add(shape);
        }   
        
        // so we evaluate the initial population
        while(evaluated < initialNumShapeletsPerSeries){
            population = evolvePopulation(timeSeries, population, checkCandidate);
        }
        

        return (ArrayList<Shapelet>) population;
    }
    
    private Pair<Integer, Integer> createRandomShapelet(double[] series){
        int numLengths = maxShapeletLength - minShapeletLength; //want max value to be inclusive.
        int length = random.nextInt(numLengths) + minShapeletLength; //offset the index by the min value.
        int position  = random.nextInt(series.length + 1 - length); // can only have valid start positions based on the length. (numAtts-1)-l+1
        return new Pair<>(length, position);
    }
    
    private static final double mutationRate = 0.015;
    private static final int tournamentSize = 5;
    private static final boolean elitism = true;
    
    private List<Shapelet> evolvePopulation(Instance timeSeries, List<Shapelet> shapesIn, ShapeletSearch.ProcessCandidate checkCandidate){
        List<Shapelet> newPopulation = new ArrayList<>();
        List<Pair<Integer, Integer>> populationToBe = new ArrayList<>();

        // Keep our best individual
        if (elitism) {
            newPopulation.add(getBestShapelet(shapesIn));
        }

        // Crossover population
        int elitismOffset = elitism ? 1 : 0;

        // Loop over the population size and create new individuals with
        // crossover
        for (int i = elitismOffset; i < shapesIn.size(); i++) {
            Shapelet indiv1 = tournamentSelection(shapesIn);
            Shapelet indiv2 = tournamentSelection(shapesIn);
            Pair<Integer, Integer> crossed = crossOver(indiv1, indiv2);
            populationToBe.add(crossed);
        }

        double[] series = timeSeries.toDoubleArray();
        
        // Mutate population
        for (Pair<Integer, Integer> populationToBe1 : populationToBe) {
            mutate(populationToBe1);
            
            //check it's valid. PURGE THE MUTANT! Replace with random valid replacement.
            if(!validMutation(populationToBe1)){
                Pair<Integer, Integer> pair = createRandomShapelet(series);
                populationToBe1 = pair;
            }
            
            Shapelet sh = checkCandidate.process(timeSeries, populationToBe1.var2, populationToBe1.var1);
            evaluated++;
            if(sh != null)
            newPopulation.add(sh);
        }
        
        return newPopulation;
    }
    
    private Shapelet getBestShapelet(List<Shapelet> shapes){
        Shapelet bsf = shapes.get(0);
        for(Shapelet s : shapes){
            if(s.getQualityValue() > bsf.getQualityValue())
                bsf = s;
        }
        return bsf;
    }
    
    private void mutate(Pair<Integer, Integer> shape){
        //random length or position mutation.
        
        if(random.nextDouble() <= mutationRate){
            //mutate length by + or - 1
            shape.var1+= random.nextBoolean() ? 1 : -1;
        }
        
        if(random.nextDouble() <= mutationRate){
            //mutate position by + or - 1
            shape.var2+= random.nextBoolean() ? 1 : -1;
        }
    }
    
    private boolean validMutation(Pair<Integer, Integer> mutant){
        int newLen = mutant.var1;
        int newPos = mutant.var2;
        int m = inputData.numAttributes() -1;
                
        return !(newLen < minShapeletLength || //don't allow length to be less than minShapeletLength. 
           newLen > maxShapeletLength || //don't allow length to be more than maxShapeletLength.
           newPos < 0                 || //don't allow position to be less than 0.               
           newPos >= (m-newLen+1));       //don't allow position to be greater than m-l+1.
    }
    
    private Pair<Integer, Integer> crossOver(Shapelet shape1, Shapelet shape2){
        //pair should be length, position.
        //take the shapelets and make them breeeeeed.
        int length, position;
        length = random.nextBoolean() ?  shape1.length : shape2.length;
        position = random.nextBoolean() ? shape1.startPos : shape2.startPos;
        return new Pair<>(length, position);
    }
    
    private Shapelet tournamentSelection(List<Shapelet> shapes){
        //create random list of shapelets and battle them off.
       // Create a tournament population
        List<Shapelet> tournament = new ArrayList<>();
        // For each place in the tournament get a random individual
        for (int i = 0; i < tournamentSize; i++) {
            int randomId = (int) (Math.random() * shapes.size());
            tournament.add(shapes.get(randomId));
        }
        
        // Get the fittest
        return getBestShapelet(tournament);
    }
}
