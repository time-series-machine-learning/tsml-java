package core;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.LineNumberReader;

import org.apache.commons.lang3.time.DurationFormatUtils;
import datasets.ListDataset;
import util.PrintUtilities;

/**
 * @author shifaz
 * @email ahmed.shifaz@monash.edu
 */

public class CSVReader {

    //helper function to assist memory allocation
    public static int[] getFileInformation(String fileName, boolean hasHeader, String separator) throws IOException {
        String line = null;
        String[] line_array = null;
        int[] file_info = new int[2];

        try (FileReader input = new FileReader(fileName);
             LineNumberReader lineNumberReader = new LineNumberReader(input)) {
            boolean length_check = true;

            while ((line = lineNumberReader.readLine()) != null) {
                if (length_check) {
                    length_check = false;
                    line_array = line.split(separator);
                }
//	            System.out.println("Line " + lineNumberReader.getLineNumber() + ": " + line);
            }
            //this output array contains file information
            if (hasHeader) {
                //number of rows;
                file_info[0] = lineNumberReader.getLineNumber() ==
                        0 ? lineNumberReader.getLineNumber() : lineNumberReader.getLineNumber() - 1;
            } else {
                //number of rows;
                file_info[0] = lineNumberReader.getLineNumber();
            }

            file_info[1] = line_array.length;  //number of columns;
        }

        return file_info;
    }


    public static ListDataset readCSVToListDataset(String fileName, boolean hasHeader, boolean targetColumnIsFirst, String separator) {
        String line = "";
        int i = 0;
        long start, end, elapsed;
        int[] file_info;
        ListDataset dataset = null;
        long used_mem;
        String[] line_array = null;
        double[] tmp;
        Double label;
        File f = new File(fileName);

        try (BufferedReader br = new BufferedReader(new FileReader(fileName))) {
            System.out.print("reading file [" + f.getName() + "]:");
            start = System.nanoTime();

            //useful for reading large files;
            file_info = getFileInformation(fileName, hasHeader, separator); //0=> no. of rows 1=> no. columns
            int expected_size = file_info[0];
            int data_length = file_info[1] - 1;  //-1 to exclude target the column

            dataset = new ListDataset(expected_size, data_length);

            while ((line = br.readLine()) != null) {
                // use comma as separator
                line_array = line.split(separator);

//                System.out.println("Line  " + i + " , Class=[" + line_array[0] + "] length= " + line_array.length);

                //allocate new memory every turn
                tmp = new double[data_length];

                if (targetColumnIsFirst) {
                    for (int j = 1; j <= data_length; j++) {
                        tmp[j - 1] = Double.parseDouble(line_array[j]);
                    }
                    label = Double.parseDouble(line_array[0]);
                } else {
                    //assume target is the last column
                    int j;
                    for (j = 0; j < data_length; j++) {
                        tmp[j] = Double.parseDouble(line_array[j]);
                    }
                    label = Double.parseDouble(line_array[j]);
                }

                dataset.add(label.intValue(), tmp);

                i++;
                if (i % 1000 == 0) {
                    if (i % 100000 == 0) {
                        System.out.print("\n");
                        if (i % 1000000 == 0) {
                            used_mem = AppContext.runtime.totalMemory() - AppContext.runtime.freeMemory();
                            System.out.print(i + ":" + used_mem / 1024 / 1024 + "mb\n");
                        } else {

                        }
                    } else {
                        System.out.print(".");
                    }
                }

            }
            end = System.nanoTime();
            elapsed = end - start;
            String time_duration = DurationFormatUtils.formatDuration((long) (elapsed / 1e6), "H:m:s.SSS");
            System.out.println("finished in " + time_duration);

        } catch (IOException e) {
//            e.printStackTrace();
            PrintUtilities.abort(e);
        }
        return dataset;
    }

}
