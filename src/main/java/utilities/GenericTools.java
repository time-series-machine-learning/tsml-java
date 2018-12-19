/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package utilities;

import java.io.File;
import java.io.FileNotFoundException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Formatter;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.Scanner;

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
}
