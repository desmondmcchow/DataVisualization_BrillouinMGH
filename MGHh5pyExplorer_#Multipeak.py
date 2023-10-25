import tkinter as tk
from tkinter import filedialog

import h5py
import h5py.defs
import h5py.utils
import h5py.h5ac
import h5py._proxy

import numpy as np

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import gridspec

from subroutine import LorentzianSpecCF_FixedNumPeaks_LRSep, LorentzianCalibCF, LinearCalibrationCF, ContourMapData, iSpectrumImagePosition

#import matplotlib.pyplot as plt

class HDF5Explorer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("HDF5 Explorer")
        self.filepath = None  # Initialize filepath as None
        self.maxsize(800*10, 600*10)

        self.open_button = tk.Button(self, text="Open HDF5 File", command=self.open_hdf5_file)
        self.open_button.grid(row=0, column=0, padx=10, pady=10)

        # Create the groups label and text area
        self.groups_label = tk.Label(text="Groups:")
        self.groups_label.grid(row=1, column=0, padx=10, pady=10)
        self.groups_text = tk.Text(height=1, width=20)
        self.groups_text.grid(row=2, column=0, padx=10, pady=10)

        self.display_ds_button = tk.Button(self, text="Display Datasets", command=self.display_datasets)
        self.display_ds_button.grid(row=3, column=0, padx=10, pady=10)

        # Create the dataset label and list area
        self.dataset_label = tk.Label(text="Datasets:")
        self.dataset_label.grid(row=4, column=0, padx=10, pady=10)
        self.dataset_listbox = tk.Listbox(self)
        self.dataset_listbox.grid(row=5, column=0, padx=10, pady=10)
        
        self.read_button = tk.Button(self, text="Read", command=self.read_data)
        self.read_button.grid(row=6, column=0, padx=10, pady=10)

        # Create the properties label and text area
        self.properties_label = tk.Label(text="Properties:")
        self.properties_label.grid(row=7, column=0, padx=10, pady=10)
        self.properties_text = tk.Text(height=20, width=20)
        self.properties_text.grid(row=8, column=0, padx=10, pady=10)
        
        # Create the Spectrum toggle button
        self.spectoggle_button = tk.Button(self, text="L", command=self.toggle_spec)
        self.spectoggle_button.grid(row=0, column=4, padx=10, pady=10)
        self.toggle_status = False
        
        # Create the reduce z step button
        self.reduceZstep_button = tk.Button(self, text="<", command=self.reduce_zstep)
        self.reduceZstep_button.grid(row=0, column=1, padx=10, pady=10)
        
        # Create the increase z step button
        self.increaseZstep_button = tk.Button(self, text=">", command=self.increase_zstep)
        self.increaseZstep_button.grid(row=0, column=3, padx=10, pady=10)
        self.Zstep = 0
        
        # Create z step label
        self.Zstep_label = tk.Label(text="%.f" %(self.Zstep))
        self.Zstep_label.grid(row=0, column=2, padx=10, pady=10)

        # Create the set limit input
        self.maplimit_text = tk.Text(height=3, width=25)
        self.maplimit_text.grid(row=0, column=5, padx=10, pady=10)
        self.maplimit_text.insert(tk.END, "Frequency:%.f,%.f\nLinewidth:%.f,%.f\nAmplitude:%.f,%.f" %(0,0,0,0,0,0))
        self.freq_lim = [0,0]
        self.line_lim = [0,0]
        self.ampl_lim = [0,0]

        # Create the set limit button
        self.setmaplimit_button = tk.Button(self, text="Limit OFF", command=self.set_maplimit)
        self.setmaplimit_button.grid(row=0, column=6, padx=10, pady=10)
        self.setmaplimit_status = False
        
        # Create the save data button
        self.savedata_button = tk.Button(self, text="Save Data", command=self.save_data)
        self.savedata_button.grid(row=1, column=5, padx=10, pady=10)
        self.savedata_status = False
        
        # Create the save image button
        self.saveimage_button = tk.Button(self, text="Save Image", command=self.save_image)
        self.saveimage_button.grid(row=1, column=6, padx=10, pady=10)
        self.saveimage_status = False

        # Create the Contour Plot Area
        self.fig = Figure(figsize=(7, 6), dpi=100) 
        
        self.ax1 = self.fig.add_subplot(221)
        self.ax2 = self.fig.add_subplot(222)
        self.ax3 = self.fig.add_subplot(223)
        self.ax4 = self.fig.add_subplot(224)
        
        self.ax1.set_title("Frequency", fontsize=8)
        self.ax2.set_title("Linewidth", fontsize=8)
        self.ax3.set_title("Amplitude", fontsize=8)
        self.ax4.set_title("BrightField", fontsize=8)
        
        self.fig.tight_layout(pad=2, w_pad=1, h_pad=1)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().grid(row=2, rowspan=7, column=1, columnspan=18, padx=10, pady=10)

        # Create the move left in spectrum list button
        self.moveleft_button = tk.Button(self, text="<(L)", command=self.moveleftspec)
        self.moveleft_button.grid(row=0, column=19, padx=10, pady=10)

        # Create the move right in spectrum list button
        self.moveright_button = tk.Button(self, text=">(R)", command=self.moverightspec)
        self.moveright_button.grid(row=0, column=20, padx=10, pady=10)
        
        # Create the move up in spectrum list button
        self.moveup_button = tk.Button(self, text="^(U)", command=self.moveupspec)
        self.moveup_button.grid(row=0, column=21, padx=10, pady=10)

        # Create the move down in spectrum list button
        self.movedown_button = tk.Button(self, text="V(D)", command=self.movedownspec)
        self.movedown_button.grid(row=0, column=22, padx=10, pady=10)
        
        # Create the save spectrum button
        self.savespectrum_button = tk.Button(self, text="Save Spectrum", command=self.save_spectrum)
        self.savespectrum_button.grid(row=1, column=23, padx=10, pady=10)
        self.savespectrum_status = False
        
        self.iSpectrum = 1
        
        # Create spectrum index text area
        self.iSpectrum_text = tk.Text(height=1, width=5)
        self.iSpectrum_text.grid(row=0, column=23, padx=10, pady=10)
        self.iSpectrum_text.insert(tk.END, "%.f" %(self.iSpectrum))
        
        # Create mark point in contour button
        self.markpoint_button = tk.Button(self, text="Unmarked", command=self.markpoint)
        self.markpoint_button.grid(row=0, column=24, padx=10, pady=10)
        self.markpoint_stat = False
        
        # Create the spectrum Plot Area
        self.figspec = Figure(figsize=(5, 6), dpi=100) 
        
        self.axspec1 = self.figspec.add_subplot(311)
        self.axspec2 = self.figspec.add_subplot(312)
        self.axspec3 = self.figspec.add_subplot(313)
        
        self.axspec1.set_title("Spectrum (L)", fontsize=8)
        self.axspec2.set_title("Spectrum (R)", fontsize=8)
        
        self.axspec1.set_xlabel("Frequency (GHz)", fontsize=8)
        self.axspec1.set_ylabel("Amplitude", fontsize=8)
        self.axspec2.set_xlabel("Frequency (GHz)", fontsize=8)
        self.axspec2.set_ylabel("Amplitude", fontsize=8)
        
        self.figspec.tight_layout(pad=2, w_pad=1, h_pad=1)

        self.canvasspec = FigureCanvasTkAgg(self.figspec, master=self)
        self.canvasspec.get_tk_widget().grid(row=2, rowspan=7, column=19, columnspan=16, padx=10, pady=10)
        

    def open_hdf5_file(self):
        self.filepath = filedialog.askopenfilename(filetypes=[("HDF5 files", "*.hdf5")])
        if self.filepath:
            try:
                # Open the HDF5 file
                file = h5py.File(self.filepath, 'r')

                # Get the list of groups in the HDF5 file
                groups = list(file.keys())

                self.groups_text.delete(1.0, tk.END)
                self.groups_text.insert(tk.END, "\n".join(groups))

                # Close the file
                file.close()

            except Exception as e:
                # Display an error message if there is any issue opening the file
                tk.messagebox.showerror("Error", str(e))

    def display_datasets(self):
        try:
            # Open the HDF5 file
            file = h5py.File(self.filepath, 'r')

            # Get the datasets within the selected group
            datasets = list(file['Exp_0'].keys())

            # Clear the listbox and insert the datasets
            self.dataset_listbox.delete(0, tk.END)
            for dataset in datasets:
                self.dataset_listbox.insert(tk.END, dataset)

            # Enable the dataset selection
            self.dataset_listbox.config(state=tk.NORMAL)

            # Close the file
            file.close()

        except Exception as e:
            # Display an error message if there is any issue opening the file or accessing the group
            tk.messagebox.showerror("Error", str(e))

    def read_data(self):
        selected_dataset = self.dataset_listbox.get(self.dataset_listbox.curselection())
        try:
            # Open the HDF5 file
            file = h5py.File(self.filepath, 'r')

            # Reading data
            motorcoords= file['Exp_0'][selected_dataset]['MotorCoords'][()]
            calibration= file['Exp_0'][selected_dataset]['CalFreq'][()]
            spectra= file['Exp_0'][selected_dataset]['SpecList'][()]
            
            # Interating quantities
            NData = np.size(motorcoords,0) # Number of scanning data points
            NSpec = int(np.size(spectra,0)) # Total number of spectra including calibration 
            NCalib = np.size(calibration,1) # Number of calibration points per X line
            NScanLine = np.size(calibration,0) # Number of scanning line           
            NXLine = int(np.size(spectra,0)/np.size(calibration,0)) # Number of points per X line including calibration
            NXElem = int(NXLine-np.size(calibration,1)) # Number of points per X line without calibration
            NHalfSpec = int(np.floor(np.size(spectra,1)/2)) # 1/2 Number of sampling points per spectrum on EMCCD 
            
            # Image resolution and range
            Res=[np.max(np.diff(motorcoords[:,0])),
                 np.max(np.diff(motorcoords[:,1])),
                 np.max(np.diff(motorcoords[:,2]))]
            Res = [1 if value == 0 else value for value in Res]
            Ran=[np.max(motorcoords[:,0])-np.min(motorcoords[:,0]),
                 np.max(motorcoords[:,1])-np.min(motorcoords[:,1]),
                 np.max(motorcoords[:,2])-np.min(motorcoords[:,2])]
            
            # Zero center motor coordinates
            ZCmotorcoods=np.zeros((NData,3))
            for i in range(3):
                ZCmotorcoods[:,i]=motorcoords[:,i]-np.mean(motorcoords[:,i])
            
            # Index coordinates
            iMotorcoods=np.zeros((NData,3))
            for i in range(3):
                iMotorcoods[:,i]=(ZCmotorcoods[:,i]-np.min(ZCmotorcoods[:,i]))/Res[i]
            
            # Number of motor steps
            motorsteps=[np.max(iMotorcoods[:,0])+1,
                        np.max(iMotorcoods[:,1])+1,
                        np.max(iMotorcoods[:,2])+1]
            
            # Calibration steps and frequencies
            fCalib=calibration[0,:]
            
            # Index of data spectra and calibration
            iSpecList = np.arange(int(NSpec-NScanLine*NCalib))
            iSpecList = iSpecList + np.floor(iSpecList/np.size(iSpecList)*NScanLine)*NCalib
            iCalibList = np.arange(int(NScanLine*NCalib))
            iCalibList = iCalibList + (np.floor(iCalibList/np.size(iCalibList)*NScanLine)+1)*NXElem
            
            print("\n\nCalculating Parameters of Calibration Spectra.")
            CalibSpecList = np.zeros([np.size(iCalibList),np.size(spectra,1)])
            for i in range(np.size(iCalibList)):
                CalibSpecList[i,:] = spectra[int(iCalibList[i]),:]
            iCalibParams = LorentzianCalibCF(CalibSpecList)
            
            # Linear curve fitting for obtaining frequency axes m and c
            fCalibParams = LinearCalibrationCF(calibration, iCalibParams)

            print("\n\nCalculating Parameters of Data Spectra.")
            DataSpecList = np.zeros([np.size(iSpecList),np.size(spectra,1)])
            for i in range(np.size(iSpecList)):
                DataSpecList[i,:] = spectra[int(iSpecList[i]),:]
            iDataParamsl, iDataParamsr, yFitDatal, yFitDatar, xfreql, xfreqr, NfitPeak = LorentzianSpecCF_FixedNumPeaks_LRSep(DataSpecList, fCalibParams, 3)
            
            # Statistics
            freqBril_mean, freqBrir_mean = np.mean(iDataParamsl[:,0]), np.mean(iDataParamsr[:,0])
            lwBril_mean, lwBrir_mean = np.mean(iDataParamsl[:,1]), np.mean(iDataParamsr[:,1])
            ampBril_mean, ampBrir_mean = np.mean(iDataParamsl[:,2]), np.mean(iDataParamsr[:,2])
            freqBril_std, freqBrir_std = np.std(iDataParamsl[:,0]), np.std(iDataParamsr[:,0])
            lwBril_std, lwBrir_std = np.std(iDataParamsl[:,1]), np.std(iDataParamsr[:,1]),
            ampBril_std, ampBrir_std = np.std(iDataParamsl[:,2]), np.std(iDataParamsr[:,2])
            
            stat_mean = np.array([[freqBril_mean, freqBrir_mean],[lwBril_mean, lwBrir_mean],[ampBril_mean, ampBrir_mean]])
            stat_std = np.array([[freqBril_std, freqBrir_std],[lwBril_std, lwBrir_std],[ampBril_std, ampBrir_std]])
            
            # Average Max & Min
            maxxfreql, minxfreql = np.zeros(NData), np.zeros(NData)
            maxxfreqr, minxfreqr = np.zeros(NData), np.zeros(NData)
            maxDatal, minDatal = np.zeros(NData), np.zeros(NData)
            maxDatar, minDatar = np.zeros(NData), np.zeros(NData)
            
            for i in range(NData):
                maxxfreql[i], minxfreql[i] = np.max(xfreql[i,:]), np.min(xfreql[i,:])
                maxxfreqr[i], minxfreqr[i] = np.max(xfreqr[i,:]), np.min(xfreqr[i,:])
                maxDatal[i], minDatal[i] = np.max(DataSpecList[i,0:NHalfSpec]), np.min(DataSpecList[i,0:NHalfSpec])
                maxDatar[i], minDatar[i] = np.max(DataSpecList[i,NHalfSpec:int(2*NHalfSpec)]), np.min(DataSpecList[i,NHalfSpec:int(2*NHalfSpec)])
            
            stat_minmaxfreq = np.array([[np.min(minxfreql), np.min(minxfreqr)],[np.max(maxxfreql), np.max(maxxfreqr)]]) 
            stat_minmaxdata = np.array([[np.min(minDatal), np.min(minDatar)],[np.max(maxDatal), np.max(maxDatar)]])
            
            # Saving selected brightfield data
            brightfield = file['Exp_0'][selected_dataset]['BrightfieldImage'][0]
            brightfield = np.rot90(brightfield)
            
            # Close the file
            file.close()
            
            # Creating global parameters
            self.NDataGlo=NData
            self.NHalfSpecGlo=NHalfSpec
            
            self.selected_datasetGlo=selected_dataset
            self.fCalibPointsGlo=fCalib
            self.fCalibParamsGlo=fCalibParams
            self.ResGlo=Res
            self.iMotorcoodsGlo=iMotorcoods
            self.ZCmotorcoodsGlo=ZCmotorcoods
            self.motorstepsGlo=motorsteps
            
            self.iSpecListGlo = iSpecList
            self.iCalibListGlo = iCalibList
            
            self.DataSpecListGlo=DataSpecList
            self.iDataParamslGlo=iDataParamsl
            self.iDataParamsrGlo=iDataParamsr
            self.yFitDatalGlo=yFitDatal
            self.yFitDatarGlo=yFitDatar
            self.xfreqlGlo=xfreql
            self.xfreqrGlo=xfreqr
            self.NfitPeakGlo=NfitPeak
            
            self.stat_meanGlo=stat_mean
            self.stat_stdGlo=stat_std
            self.stat_minmaxfreqGlo=stat_minmaxfreq
            self.stat_minmaxdataGlo=stat_minmaxdata
            
            self.brightfieldSelGlo=brightfield
            
            # Display Scan Data Summary
            self.properties_text.delete(1.0, tk.END)
            self.properties_text.insert(tk.END, "Data Points: %.f" %(NData)) 
            self.properties_text.insert(tk.END, "\n x>\n   Res: %.2f\n   Range: %.2f\n   Steps: %.0f" %(Res[0],Ran[0],motorsteps[0]))
            self.properties_text.insert(tk.END, "\n y>\n   Res: %.2f\n   Range: %.2f\n   Steps: %.0f" %(Res[1],Ran[1],motorsteps[1])) 
            self.properties_text.insert(tk.END, "\n z>\n   Res: %.2f\n   Range: %.2f\n   Steps: %.0f" %(Res[2],Ran[2],motorsteps[2])) 
            self.properties_text.insert(tk.END, "\nCalibration: %.f\n" %(NCalib))
            self.properties_text.insert(tk.END, ",".join(str(x) for x in fCalib))
            
        except Exception as e:
            # Display an error message if there is any issue opening the file or accessing the group
            tk.messagebox.showerror("Error", str(e))
        
        print("\n\n[Complete]")
        self.plot_contour()
        self.plot_spectra()
        
    def plot_contour(self):
         
        self.fig.clear()
        
        spec = gridspec.GridSpec(ncols=2, nrows=2, hspace=0.3, wspace=0.3)
        
        self.ax1 = self.fig.add_subplot(spec[0])
        self.ax2 = self.fig.add_subplot(spec[1])
        self.ax3 = self.fig.add_subplot(spec[2])
        self.ax4 = self.fig.add_subplot(spec[3])

        iMotorcoods = self.iMotorcoodsGlo        
        
        # Generate the contour map data
        if (self.toggle_status == False):
            
            # Left of spectrum 
            freqBri = self.iDataParamslGlo[:,0]
            Map_freqBri = ContourMapData(freqBri, iMotorcoods)
        
            lwBri = self.iDataParamslGlo[:,1]
            Map_lwBri = ContourMapData(lwBri, iMotorcoods)
        
            ampBri = self.iDataParamslGlo[:,2]
            Map_ampBri = ContourMapData(ampBri, iMotorcoods)
            
            freqBri_mean, lwBri_mean, ampBri_mean = self.stat_meanGlo[:,0]
            freqBri_std, lwBri_std, ampBri_std = self.stat_stdGlo[:,0]
            
            self.ax1.set_title("Frequency (L) Z=%.f" %(self.Zstep), fontsize=8)
            self.ax2.set_title("Linewidth (L) Z=%.f" %(self.Zstep), fontsize=8)
            self.ax3.set_title("Amplitude (L) Z=%.f" %(self.Zstep), fontsize=8)
            
        else:
            
            # Right of spectrum 
            freqBri = self.iDataParamsrGlo[:,0]
            Map_freqBri = ContourMapData(freqBri, iMotorcoods)
        
            lwBri = self.iDataParamsrGlo[:,1]
            Map_lwBri = ContourMapData(lwBri, iMotorcoods)
        
            ampBri = self.iDataParamsrGlo[:,2]
            Map_ampBri = ContourMapData(ampBri, iMotorcoods)
            
            freqBri_mean, lwBri_mean, ampBri_mean = self.stat_meanGlo[:,1]
            freqBri_std, lwBri_std, ampBri_std = self.stat_stdGlo[:,1]
            
            self.ax1.set_title("Frequency (R) Z=%.f" %(self.Zstep), fontsize=8)
            self.ax2.set_title("Linewidth (R) Z=%.f" %(self.Zstep), fontsize=8)
            self.ax3.set_title("Amplitude (R) Z=%.f" %(self.Zstep), fontsize=8)
        
        # calculate the limit values for display
        freqBri_lim = [np.round((freqBri_mean-2.5*freqBri_std)*10)/10, np.round((freqBri_mean+2.5*freqBri_std)*10)/10]
        lwBri_lim = [np.round((lwBri_mean-2.5*lwBri_std)*10)/10, np.round((lwBri_mean+2.5*lwBri_std)*10)/10]
        ampBri_lim = [np.round((ampBri_mean-2.5*ampBri_std)), np.round((ampBri_mean+2.5*ampBri_std))]
        if (freqBri_lim[0] < 0): freqBri_lim[0] = 0
        if (lwBri_lim[0] < 0): lwBri_lim[0] = 0
        if (ampBri_lim[0] < 0): ampBri_lim[0] = 0
        
        xbound = [np.min(self.ZCmotorcoodsGlo[:,0])-self.ResGlo[0]/2, np.max(self.ZCmotorcoodsGlo[:,0])+self.ResGlo[0]/2]
        ybound = [np.max(self.ZCmotorcoodsGlo[:,1])+self.ResGlo[1]/2, np.min(self.ZCmotorcoodsGlo[:,1])-self.ResGlo[1]/2]
                
        # Creating contour plot with imshow
        if (self.setmaplimit_status == True):
            im1 = self.ax1.imshow(np.transpose(Map_freqBri[:,:,int(self.Zstep)]), cmap='turbo', interpolation='none', vmin=self.freq_lim[0], vmax=self.freq_lim[1], extent=[xbound[0], xbound[1], ybound[0], ybound[1]])
            im2 = self.ax2.imshow(np.transpose(Map_lwBri[:,:,int(self.Zstep)]), cmap='viridis', interpolation='none', vmin=self.line_lim[0], vmax=self.line_lim[1], extent=[xbound[0], xbound[1], ybound[0], ybound[1]])
            im3 = self.ax3.imshow(np.transpose(Map_ampBri[:,:,int(self.Zstep)]), cmap='cividis', interpolation='none', vmin=self.ampl_lim[0], vmax=self.ampl_lim[1], extent=[xbound[0], xbound[1], ybound[0], ybound[1]])
        else: 
            self.maplimit_text.delete(1.0, tk.END)
            self.maplimit_text.insert(tk.END, "Frequency:%.1f,%.1f\nLinewidth:%.1f,%.1f\nAmplitude:%.f,%.f" %(freqBri_lim[0],freqBri_lim[1],lwBri_lim[0],lwBri_lim[1],ampBri_lim[0],ampBri_lim[1]))
            im1 = self.ax1.imshow(np.transpose(Map_freqBri[:,:,int(self.Zstep)]), cmap='turbo', interpolation='none', vmin=freqBri_lim[0], vmax=freqBri_lim[1], extent=[xbound[0], xbound[1], ybound[0], ybound[1]])
            im2 = self.ax2.imshow(np.transpose(Map_lwBri[:,:,int(self.Zstep)]), cmap='viridis', interpolation='none', vmin=lwBri_lim[0], vmax=lwBri_lim[1], extent=[xbound[0], xbound[1], ybound[0], ybound[1]])
            im3 = self.ax3.imshow(np.transpose(Map_ampBri[:,:,int(self.Zstep)]), cmap='cividis', interpolation='none', vmin=ampBri_lim[0], vmax=ampBri_lim[1], extent=[xbound[0], xbound[1], ybound[0], ybound[1]])            
        
        self.ax1.set_xlabel("x (um)", fontsize=8)
        self.ax1.set_ylabel("y (um)", fontsize=8)
        self.ax2.set_xlabel("x (um)", fontsize=8)
        self.ax2.set_ylabel("y (um)", fontsize=8)
        self.ax3.set_xlabel("x (um)", fontsize=8)
        self.ax3.set_ylabel("y (um)", fontsize=8)
        
        self.ax1.xaxis.set_tick_params(labelsize=8)
        self.ax1.yaxis.set_tick_params(labelsize=8)
        self.ax2.xaxis.set_tick_params(labelsize=8)
        self.ax2.yaxis.set_tick_params(labelsize=8)
        self.ax3.xaxis.set_tick_params(labelsize=8)
        self.ax3.yaxis.set_tick_params(labelsize=8)
        
        divider = make_axes_locatable(self.ax1)
        cax1 = divider.append_axes("right", size="5%", pad=0.1)
        cbar1 = self.fig.colorbar(im1, ax=self.ax1, cax=cax1)
        cbar1.set_label('Frequency (GHz)', fontsize=8)
        cbar1.ax.tick_params(labelsize=8)
        
        divider = make_axes_locatable(self.ax2)
        cax2 = divider.append_axes("right", size="5%", pad=0.1)
        cbar2 = self.fig.colorbar(im2, ax=self.ax2, cax=cax2)
        cbar2.set_label('Linewidth (GHz)', fontsize=8) 
        cbar2.ax.tick_params(labelsize=8)
        
        divider = make_axes_locatable(self.ax3)
        cax3 = divider.append_axes("right", size="5%", pad=0.1)
        cbar3 = self.fig.colorbar(im3, ax=self.ax3, cax=cax3)
        cbar3.set_label('Amplitude', fontsize=8) 
        cbar3.ax.tick_params(labelsize=8)
        
        # Creating mark point
        if(self.markpoint_stat == True):
            x_point=(iMotorcoods[int(self.iSpectrum),0]-np.max(iMotorcoods[:,0])/2)*self.ResGlo[0]
            y_point=(iMotorcoods[int(self.iSpectrum),1]-np.max(iMotorcoods[:,1])/2)*self.ResGlo[1]
            point = [x_point, y_point]
            self.ax1.annotate('', xy=point, xytext=(15, -15), textcoords='offset points', arrowprops=dict(facecolor='white', shrink=0.001),)
            self.ax2.annotate('', xy=point, xytext=(15, -15), textcoords='offset points', arrowprops=dict(facecolor='white', shrink=0.001),)
            self.ax3.annotate('', xy=point, xytext=(15, -15), textcoords='offset points', arrowprops=dict(facecolor='white', shrink=0.001),)
        
        # Read the brightfield data
        selected_dataset = self.selected_datasetGlo
        try:
            # Open the HDF5 file
            file = h5py.File(self.filepath, 'r')

            # Reading data
            brightfield = file['Exp_0'][selected_dataset]['BrightfieldImage'][int(self.Zstep)]
            brightfield = np.rot90(brightfield)
            self.ax4.imshow(brightfield, cmap='gray', interpolation='none')
            self.ax4.set_xticks([])
            self.ax4.set_yticks([])
            
            # Close the file
            file.close()
            
        except Exception as e:
            # Display an error message if there is any issue opening the file or accessing the group
            tk.messagebox.showerror("Error", str(e))
            
        self.canvas.draw()
        
        if (self.saveimage_status == True):
            saveimage_file_path = self.filepath.replace(".hdf5","") + "_" + self.selected_datasetGlo.replace("_","") + "z" + str(int(self.Zstep)) + '_BParamMapFig.png'
            self.fig.savefig(saveimage_file_path, dpi=300)
            print("Image Plot File Path:" + saveimage_file_path)
    
    def plot_spectra(self):

            self.figspec.clear()
            
            spec = gridspec.GridSpec(ncols=1, nrows=3, hspace=1, height_ratios=[1, 1, 0.5])
            
            self.axspec1 = self.figspec.add_subplot(spec[0])
            self.axspec2 = self.figspec.add_subplot(spec[1])
            self.axspec3 = self.figspec.add_subplot(spec[2])
            
            self.axspec1.set_title("Spectrum (L) Spec = %.f" %(int(self.iSpectrum)), fontsize=8)
            self.axspec2.set_title("Spectrum (R) Spec = %.f" %(int(self.iSpectrum)), fontsize=8)
        
            self.axspec1.set_xlabel("Frequency (GHz)", fontsize=8)
            self.axspec1.set_ylabel("Amplitude", fontsize=8)
            self.axspec2.set_xlabel("Frequency (GHz)", fontsize=8)
            self.axspec2.set_ylabel("Amplitude", fontsize=8)

            self.axspec1.xaxis.set_tick_params(labelsize=8)
            self.axspec1.yaxis.set_tick_params(labelsize=8)
            self.axspec2.xaxis.set_tick_params(labelsize=8)
            self.axspec2.yaxis.set_tick_params(labelsize=8)
            
            NSpecHalf = np.size(self.xfreqlGlo,1)
  
            # Left spectrum
            freql = self.xfreqlGlo[int(self.iSpectrum),:]
            ydatal = self.DataSpecListGlo[int(self.iSpectrum),0:NSpecHalf]
            yfitl = self.yFitDatalGlo[int(self.iSpectrum),:]
            
            xliml = [0, np.round(self.stat_minmaxfreqGlo[1,0]*10)/10]
            yliml = [np.floor(self.stat_minmaxdataGlo[0,0]/100)*100, np.ceil(self.stat_minmaxdataGlo[1,0]/100)*100]
            
            self.axspec1.scatter(freql, ydatal, s=20, label='Data')
            self.axspec1.plot(freql, yfitl, color='red', label='Fit')
            self.axspec1.set_xlim([xliml[0], xliml[1]])
            self.axspec1.set_ylim([yliml[0], yliml[1]])
            self.axspec1.set_aspect(0.5*abs(np.diff(xliml)/np.diff(yliml)))
            self.axspec1.set_anchor('W')
            
            text_content01 = ""
            for i in range(int(self.NfitPeakGlo[self.iSpectrum,0])):
                text_content01 += "Peak(%.f)\nFrequency: %.3f GHz\nLinewidth: %.3f GHz\nAmplitude: %.f\n\n" %(
                    i+1,
                    self.iDataParamslGlo[int(self.iSpectrum),int(3*i+0)],
                    self.iDataParamslGlo[int(self.iSpectrum),int(3*i+1)],
                    self.iDataParamslGlo[int(self.iSpectrum),int(3*i+2)])
            self.axspec1.annotate(text_content01, xy=(0, 0), xytext=(1.05, -0.2), xycoords='axes fraction', fontsize=8)
           
            # Right spectrum
            freqr = np.flip(self.xfreqrGlo[int(self.iSpectrum),:])
            ydatar = np.flip(self.DataSpecListGlo[int(self.iSpectrum),int(NSpecHalf):int(2*NSpecHalf)])
            yfitr = np.flip(self.yFitDatarGlo[int(self.iSpectrum),:])
            
            xlimr = [0, np.round(self.stat_minmaxfreqGlo[1,1]*10)/10]
            ylimr = [np.floor(self.stat_minmaxdataGlo[0,1]/100)*100, np.ceil(self.stat_minmaxdataGlo[1,1]/100)*100]
                
            self.axspec2.scatter(freqr, ydatar, s=20, label='Data')
            self.axspec2.plot(freqr, yfitr, color='red', label='Fit')
            self.axspec2.set_xlim([xlimr[0], xlimr[1]])
            self.axspec2.set_ylim([ylimr[0], ylimr[1]])
            self.axspec2.set_aspect(0.5*abs(np.diff(xlimr)/np.diff(ylimr)))
            self.axspec2.set_anchor('W')

            text_content02 = ""
            for i in range(int(self.NfitPeakGlo[self.iSpectrum,1])):
                text_content02 += "Peak(%.f)\nFrequency: %.3f GHz\nLinewidth: %.3f GHz\nAmplitude: %.f\n\n" %(
                    i+1,
                    self.iDataParamsrGlo[int(self.iSpectrum),int(3*i+0)],
                    self.iDataParamsrGlo[int(self.iSpectrum),int(3*i+1)],
                    self.iDataParamsrGlo[int(self.iSpectrum),int(3*i+2)])
            self.axspec2.annotate(text_content02, xy=(0, 0), xytext=(1.05, -0.2), xycoords='axes fraction', fontsize=8)
            
            # Read the Andor CCD data
            selected_dataset = self.selected_datasetGlo
            
            try:
                # Open the HDF5 file
                file = h5py.File(self.filepath, 'r')

                # Reading data
                andorimage= file['Exp_0'][selected_dataset]['AndorImage'][int(self.iSpecListGlo[int(self.iSpectrum)])]
                self.axspec3.imshow(andorimage, cmap='gray', interpolation='none')
                self.axspec3.set_xticks([])
                self.axspec3.set_yticks([])
                self.axspec3.set_anchor('W')
                            
                # Close the file
                file.close()
            
            except Exception as e:
                # Display an error message if there is any issue opening the file or accessing the group
                tk.messagebox.showerror("Error", str(e))
            
            self.canvasspec.draw()
            
            if (self.savespectrum_status == True):
                savespectrum_file_path = self.filepath.replace(".hdf5","") + "_" + self.selected_datasetGlo.replace("_","") + "z" + str(int(self.iSpectrum)) + '_BSpecFig.png'
                self.figspec.savefig(savespectrum_file_path, dpi=300)
                print("Spectrum Plot File Path:" + savespectrum_file_path)
            
            if(self.markpoint_stat == True):
                self.plot_contour()
    
    def toggle_spec(self):
        self.toggle_status = not(self.toggle_status)
        if(self.toggle_status == False):
            self.spectoggle_button.config(text="L")
        else:
            self.spectoggle_button.config(text="R")
        self.plot_contour()
        
    def reduce_zstep(self):
        if(self.Zstep > 0):
            self.Zstep = self.Zstep - 1
            self.Zstep_label.config(text="%.f" %(self.Zstep))
        
        self.iSpectrum = self.motorstepsGlo[0]*self.motorstepsGlo[1]*self.Zstep
        self.plot_contour()
        
        self.iSpectrum_text.delete(1.0, tk.END)
        self.iSpectrum_text.insert(tk.END, "%.f" %(self.iSpectrum))
        self.plot_spectra()
    
    def increase_zstep(self):
        if(self.Zstep < int(self.motorstepsGlo[2]-1)):
            self.Zstep = self.Zstep + 1
            self.Zstep_label.config(text="%.f" %(self.Zstep))
        
        self.iSpectrum = self.motorstepsGlo[0]*self.motorstepsGlo[1]*self.Zstep
        self.plot_contour()
        
        self.iSpectrum_text.delete(1.0, tk.END)
        self.iSpectrum_text.insert(tk.END, "%.f" %(self.iSpectrum))
        self.plot_spectra()
            
    def movedownspec(self):
        
        Mdir = [1,0]
        
        input_ispec = int(self.iSpectrum_text.get(1.0, tk.END))
        
        ispec_update = iSpectrumImagePosition(input_ispec, self.motorstepsGlo, Mdir)
        
        self.iSpectrum = int(ispec_update)       
        self.iSpectrum_text.delete(1.0, tk.END)
        self.iSpectrum_text.insert(tk.END, "%.f" %(self.iSpectrum))
        self.plot_spectra()

    def moveupspec(self):
        
        Mdir = [1,1]
        
        input_ispec = int(self.iSpectrum_text.get(1.0, tk.END))
        
        ispec_update = iSpectrumImagePosition(input_ispec, self.motorstepsGlo, Mdir)
        
        self.iSpectrum = int(ispec_update)       
        self.iSpectrum_text.delete(1.0, tk.END)
        self.iSpectrum_text.insert(tk.END, "%.f" %(self.iSpectrum))
        self.plot_spectra()
        
    def moveleftspec(self):
        
        Mdir = [0,1]
        
        input_ispec = int(self.iSpectrum_text.get(1.0, tk.END))
        
        ispec_update = iSpectrumImagePosition(input_ispec, self.motorstepsGlo, Mdir)
        
        self.iSpectrum = int(ispec_update)       
        self.iSpectrum_text.delete(1.0, tk.END)
        self.iSpectrum_text.insert(tk.END, "%.f" %(self.iSpectrum))
        self.plot_spectra()
        
    def moverightspec(self):
        
        Mdir = [0,0]
        
        input_ispec = int(self.iSpectrum_text.get(1.0, tk.END))
        
        ispec_update = iSpectrumImagePosition(input_ispec, self.motorstepsGlo, Mdir)
        
        self.iSpectrum = int(ispec_update)       
        self.iSpectrum_text.delete(1.0, tk.END)
        self.iSpectrum_text.insert(tk.END, "%.f" %(self.iSpectrum))
        self.plot_spectra()

    def markpoint(self):
        self.markpoint_stat = not(self.markpoint_stat)
        if(self.markpoint_stat == False):
            self.markpoint_button.config(text="Unmarked")
        else:
            self.markpoint_button.config(text="Marked")
            
    def save_image(self):
        self.saveimage_status=True
        self.plot_contour()
        self.saveimage_status=False
        
    def save_spectrum(self):
        self.savespectrum_status=True
        self.plot_spectra()
        self.savespectrum_status=False
        
    def save_data(self):
        self.savedata_status=True
        
        # Save the data to a CSV file
        print("\n\nSaving Data Files...\n")
        
        file_path = self.filepath.replace(".hdf5","") + "_" + self.selected_datasetGlo.replace("_","") +'_CFBrilParL.csv'  # Specify the file path and name
        with open(file_path, 'w') as csvfile:
            # Write each data row
            for i in range(self.NDataGlo):
                csvfile.write(",".join(str(x) for x in self.iDataParamslGlo[i,:]))
                csvfile.write("\n")
        print("\n1. Data File Path (L):" + file_path)
        
        file_path = self.filepath.replace(".hdf5","") + "_" + self.selected_datasetGlo.replace("_","") +'_CFBrilParR.csv'  # Specify the file path and name
        with open(file_path, 'w') as csvfile:
            # Write each data row
            for i in range(self.NDataGlo):
                csvfile.write(",".join(str(x) for x in self.iDataParamsrGlo[i,:]))
                csvfile.write("\n")            
        print("\n2. Data File Path (R):" + file_path)
        
        file_path = self.filepath.replace(".hdf5","") + "_" + self.selected_datasetGlo.replace("_","") +'_SpecListL.csv'  # Specify the file path and name
        with open(file_path, 'w') as csvfile:
            # Write each data row
            for i in range(self.NDataGlo):
                csvfile.write(",".join(str(x) for x in self.DataSpecListGlo[i,0:int(self.NHalfSpecGlo)]))
                csvfile.write("\n")            
        print("\n3. Spectrum File Path (L):" + file_path)
        
        file_path = self.filepath.replace(".hdf5","") + "_" + self.selected_datasetGlo.replace("_","") +'_SpecListR.csv'  # Specify the file path and name
        with open(file_path, 'w') as csvfile:
            # Write each data row
            for i in range(self.NDataGlo):
                csvfile.write(",".join(str(x) for x in self.DataSpecListGlo[i,int(self.NHalfSpecGlo):int(2*self.NHalfSpecGlo)]))
                csvfile.write("\n")            
        print("\n4. Spectrum File Path (R):" + file_path)
        
        file_path = self.filepath.replace(".hdf5","") + "_" + self.selected_datasetGlo.replace("_","") +'_FitSpecListL.csv'  # Specify the file path and name
        with open(file_path, 'w') as csvfile:
            # Write each data row
            for i in range(self.NDataGlo):
                csvfile.write(",".join(str(x) for x in self.yFitDatalGlo[i,:]))
                csvfile.write("\n")            
        print("\n5. Fitted Spectrum File Path (L):" + file_path)
        
        file_path = self.filepath.replace(".hdf5","") + "_" + self.selected_datasetGlo.replace("_","") +'_FitSpecListR.csv'  # Specify the file path and name
        with open(file_path, 'w') as csvfile:
            # Write each data row
            for i in range(self.NDataGlo):
                csvfile.write(",".join(str(x) for x in self.yFitDatarGlo[i,:]))
                csvfile.write("\n")            
        print("\n6. Fitted Spectrum File Path (R):" + file_path)
        
        file_path = self.filepath.replace(".hdf5","") + "_" + self.selected_datasetGlo.replace("_","") +'_xFreqListL.csv'  # Specify the file path and name
        with open(file_path, 'w') as csvfile:
            # Write each data row
            for i in range(self.NDataGlo):
                csvfile.write(",".join(str(x) for x in self.xfreqlGlo[i,:]))
                csvfile.write("\n")            
        print("\n7. Frequency Axis File Path (L):" + file_path)

        file_path = self.filepath.replace(".hdf5","") + "_" + self.selected_datasetGlo.replace("_","") +'_xFreqListR.csv'  # Specify the file path and name
        with open(file_path, 'w') as csvfile:
            # Write each data row
            for i in range(self.NDataGlo):
                csvfile.write(",".join(str(x) for x in self.xfreqrGlo[i,:]))
                csvfile.write("\n")            
        print("\n8. Frequency Axis File Path (R):" + file_path)
        
        # Saving motor coordinates data
        file_path = self.filepath.replace(".hdf5","") + "_" + self.selected_datasetGlo.replace("_","") +'_ZCMotorCoor.csv'  # Specify the file path and name
        with open(file_path, 'w') as csvfile:
            # Write the header row
            csvfile.write('x, y, z\n')
            # Write each data row
            for i in range(self.NDataGlo):
                csvfile.write(",".join(str(x) for x in self.ZCmotorcoodsGlo[i,:]))
                csvfile.write("\n")     
        print("\n9. Motor File Path:" + file_path)
            
        # Saving selected brightfield data
        file_path = self.filepath.replace(".hdf5","") + "_" + self.selected_datasetGlo.replace("_","") +'_Brightfieldz0.csv'  # Specify the file path and name
        with open(file_path, 'w') as csvfile:
            # Write each data row
            for i in range(np.size(self.brightfieldSelGlo[:,0])):
                csvfile.write(",".join(str(x) for x in self.brightfieldSelGlo[i,:]))
                csvfile.write("\n")      
        print("\n10. Brightfield File Path:" + file_path)
        
        self.savedata_status=False
        
    def set_maplimit(self):
        self.setmaplimit_status = not(self.setmaplimit_status)
        if(self.setmaplimit_status == False):
            self.setmaplimit_button.config(text="Limit OFF")
            self.setmaplimit_status = False
            self.plot_contour()
            
        else:
            self.setmaplimit_button.config(text="Limit ON")
            self.setmaplimit_status = True
            
            limit_text = self.maplimit_text.get(1.0, tk.END)
            limit_text_arr = limit_text.split("\n")
            
            limit_text_arr0 = limit_text_arr[0].split(":")
            limit_text_arr1 = limit_text_arr[1].split(":")
            limit_text_arr2 = limit_text_arr[2].split(":")
            
            freq_lim_str = limit_text_arr0[1].split(",")
            line_lim_str = limit_text_arr1[1].split(",")
            ampl_lim_str = limit_text_arr2[1].split(",")
            
            self.freq_lim = [float(freq_lim_str[0]),float(freq_lim_str[1])]
            self.line_lim = [float(line_lim_str[0]),float(line_lim_str[1])]
            self.ampl_lim = [float(ampl_lim_str[0]),float(ampl_lim_str[1])]
            
            self.plot_contour()
                        
# Create an instance of the HDF5Explorer class
explorer = HDF5Explorer()

# Start the main event loop
explorer.mainloop()