
package utilities;

import java.io.File;
import java.io.FileFilter;
import java.io.FileNotFoundException;
import java.io.FilenameFilter;
import java.io.IOException;

/**
 *
 * @author James Large (james.large@uea.ac.uk)
 */
public class FileHandlingTools {

    public static void recursiveDelete(String directory) throws IOException {
        recursiveDelete(new File(directory));
    }
    
    public static void recursiveDelete(File directory) throws IOException {
        if (directory.isDirectory()) {
          for (File subDirectory : directory.listFiles())
            recursiveDelete(subDirectory);
        }
        if (!directory.delete())
          throw new FileNotFoundException("Failed to delete file: " + directory);
    }
    
    /**
     * List the directories contained in the directory given
     */
    public static File[] listDirectories(String baseDirectory) {
        return (new File(baseDirectory)).listFiles(new FileFilter() {
            @Override
            public boolean accept(File pathname) {
                return pathname.isDirectory();
            }
        });
    }

    /**
     * List the directories contained in the directory given
     */
    public static String[] listDirectoryNames(String baseDirectory) {
        return (new File(baseDirectory)).list(new FilenameFilter() {
            @Override
            public boolean accept(File dir, String name) {
                return dir.isDirectory();
            }
        });
    }
    
     /**
     * List the files contained in the directory given
     */
    public static File[] listFiles(String baseDirectory) {
        return (new File(baseDirectory)).listFiles(new FileFilter() {
            @Override
            public boolean accept(File pathname) {
                return pathname.isFile();
            }
        });
    }
    
     /**
     * List the files contained in the directory given
     */
    public static String[] listFileNames(String baseDirectory) {
        return (new File(baseDirectory)).list(new FilenameFilter() {
            @Override
            public boolean accept(File dir, String name) {
                return dir.isFile();
            }
        });
    }
    
     /**
     * List the files contained in the directory given, that end with the given suffix (file extension, generally)
     */
    public static File[] listFilesEndingWith(String baseDirectory, String suffix) {
        return (new File(baseDirectory)).listFiles(new FileFilter() {
            @Override
            public boolean accept(File pathname) {
                return pathname.isFile() && pathname.getName().endsWith(suffix);
            }
        });
    }
    
     /**
     * List the files contained in the directory given, that end with the given suffix (file extension, generally)
     */
    public static String[] listFileNamesEndingWith(String baseDirectory, String suffix) {
        return (new File(baseDirectory)).list(new FilenameFilter() {
            @Override
            public boolean accept(File dir, String name) {
                return dir.isFile() && name.endsWith(suffix);
            }
        });
    }
    
}
