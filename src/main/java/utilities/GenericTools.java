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
package utilities;

import java.io.File;
import java.io.FileNotFoundException;
import java.text.DecimalFormat;
import java.util.*;

/**
 *
 * @author raj09hxu
 */
public class GenericTools {
    
    public static final DecimalFormat RESULTS_DECIMAL_FORMAT = new DecimalFormat("#.######");
    
    public static List<String> readFileLineByLineAsList(String filename) throws FileNotFoundException {
        Scanner filein = new Scanner(new File(filename));
        
        List<String> dsets = new ArrayList<>();
        while (filein.hasNextLine())
            dsets.add(filein.nextLine());
        
        return dsets;
    }
    
    public static String[] readFileLineByLineAsArray(String filename) throws FileNotFoundException {
        return readFileLineByLineAsList(filename).toArray(new String[] { });
    }
    
    public static double indexOfMin(double[] dist) {
        double min = dist[0];
        int minInd = 0;
        
        for (int i = 1; i < dist.length; ++i) {
            if (dist[i] < min) {
                min = dist[i];
                minInd = i;
            }
        }
        return minInd;
    }
    
    public static double min(double[] array) {
        return array[(int)indexOfMin(array)];
    }
    
    public static double indexOfMax(double[] dist) {
        double max = dist[0];
        int maxInd = 0;
        
        for (int i = 1; i < dist.length; ++i) {
            if (dist[i] > max) {
                max = dist[i];
                maxInd = i;
            }
        }
        return maxInd;
    }
    
    public static double max(double[] array) {
        return array[(int)indexOfMax(array)];
    }
    
    public static double indexOf(double[] array, double val){
        for(int i=0; i<array.length; i++)
            if(array[i] == val)
               return i;
        
        return -1;
    }
    
    public static <E> ArrayList<E> cloneArrayList(ArrayList<E> list){
        ArrayList<E> temp = new ArrayList<>();
        for (E el : list) {
            temp.add(el);
        }
        return temp;              
    }

    public static <E> ArrayList<E> twoDArrayToList(E[][] twoDArray) {
        ArrayList<E> list = new ArrayList<>();
        for (E[] array : twoDArray){
            if(array == null) continue;
            
            for(E elm : array){
                if(elm == null) continue;
                
                list.add(elm);
            }
        }
        return list;
    }
    
    //this is inclusive of the top value.
    public static int randomRange(Random rand, int min, int max){
        return rand.nextInt((max - min) + 1) + min;
    }
    
    
    public static String sprintf(String format, Object... strings){
        StringBuilder sb = new StringBuilder();
        String out;
        try (Formatter ft = new Formatter(sb, Locale.UK)) {
            ft.format(format, strings);
            out = ft.toString();
        }
        return out;
    }
    
    /**
     * assumes it's a square matrix basically. uneven inner array lengths will mess up
     */
    public static double[][] cloneAndTranspose(double[][] in) {
        double[][] out = new double[in[0].length][in.length];
        
        for (int i = 0; i < in.length; i++)
            for (int j = 0; j < in[0].length; j++)
                out[j][i] = in[i][j];
        return out;
    }

    public static class SortIndexDescending implements Comparator<Integer> {
        private double[] values;

        public SortIndexDescending(double[] values){
            this.values = values;
        }

        public Integer[] getIndicies(){
            Integer[] indicies = new Integer[values.length];
            for (int i = 0; i < values.length; i++) {
                indicies[i] = i;
            }
            return indicies;
        }

        @Override
        public int compare(Integer index1, Integer index2) {
            if (values[index2] < values[index1]){
                return -1;
            }
            else if (values[index2] > values[index1]){
                return 1;
            }
            else{
                return 0;
            }
        }
    }

    public static class SortIndexAscending implements Comparator<Integer>{
        private double[] values;

        public SortIndexAscending(double[] values){
            this.values = values;
        }

        public Integer[] getIndicies(){
            Integer[] indicies = new Integer[values.length];
            for (int i = 0; i < values.length; i++) {
                indicies[i] = i;
            }
            return indicies;
        }

        @Override
        public int compare(Integer index1, Integer index2) {
            if (values[index1] < values[index2]){
                return -1;
            }
            else if (values[index1] > values[index2]){
                return 1;
            }
            else{
                return 0;
            }
        }
    }

    public static double[] linSpace(int numValues, double min, double max){
        double[] d = new double[numValues];
        double step = (max-min)/(numValues-1);
        for (int i = 0; i < numValues; i++){
            d[i] = min + i * step;
        }
        return d;
    }
}
