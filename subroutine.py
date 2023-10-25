from scipy.optimize import curve_fit
import numpy as np

#import matplotlib.pyplot as plt

def lorentzian_double(x, x01, gamma1, A1, x02, gamma2, A2, C0):
    return (
        # Standard Lorentzian definition
        A1 * (0.5*gamma1/np.pi) / ((x - x01)**2 + (0.5*gamma1)**2) +
        A2 * (0.5*gamma2/np.pi) / ((x - x02)**2 + (0.5*gamma2)**2) + C0
    )

def lorentzian_multipeaks(x, *params):
    y = np.zeros(np.size(x))
    for i in range(0, int(len(params)-1), 3):
        center, width, amp = params[i], params[i + 1], params[i + 2]
        # Lorentzian definition from Boyd
        y += amp * (0.5*2*np.pi*width)**2 / ((2*np.pi*(center - x))**2 + (0.5*2*np.pi*width)**2 + 1e-100) 
    y +=  params[-1]
    return y

def linear(x, m, C0):
    return (
        m*x + C0
    )

def gaussian_simple(x, a, b, c):
    return (
        a*np.exp(-(x - b)**2/(2*c**2))
    )

def LorentzianSpecCF_FixedNumPeaks_LRSep(spectra, fCalibParams, NPeaks):
    
    # related total numberz
    NSpec = np.size(spectra,0)
    NSpecSam = np.size(spectra,1)
    HalfNSpecSam = int(np.floor(NSpecSam/2))
    NLine = np.size(fCalibParams,0)
    NElemX = np.floor(NSpec/NLine)
    
    # storage parameters and fitted spectrum array
    iParamsl = np.zeros([NSpec,int(3*NPeaks+1)])
    iParamsr = np.zeros([NSpec,int(3*NPeaks+1)]) 
    iyfitl = np.zeros([NSpec,HalfNSpecSam])
    iyfitr = np.zeros([NSpec,HalfNSpecSam])
    xfreql = np.zeros([NSpec,HalfNSpecSam])
    xfreqr = np.zeros([NSpec,HalfNSpecSam])
    Nfitpklist = np.zeros([NSpec,2])
    
    # calibration data
    fcalpar = np.zeros([2,2])

    
    print("\nSpectra Data Lorentzian Curve Fitting:")
    
    for i in range(NSpec):
        
        # spectrum
        y = np.zeros([2,HalfNSpecSam])
        y[0,:] = spectra[i,0:HalfNSpecSam] 
        y[1,:] = spectra[i,HalfNSpecSam:NSpecSam]
        x = np.arange(HalfNSpecSam)  

        # Perform curve fitting
        # Extract the fitted parameters > xn_fit, gamman_fit, An_fit, ... , c0

        # line number 
        iLine = int(np.floor(i/NElemX))
        
        # Retriving calibration data 
        fcalpar[0,0] = fCalibParams[iLine,0]
        fcalpar[0,1] = fCalibParams[iLine,1]
        fcalpar[1,0] = fCalibParams[iLine,2]
        fcalpar[1,1] = fCalibParams[iLine,3]

        # compute initial guess values for the number and position of peaks
        minsep = 10 # minimum peak separation  
        
        for j in range(2):
            
            params = np.zeros(int(3*NPeaks+1))
            paramssort = np.zeros(int(3*NPeaks+1))
            paramcal = np.zeros(int(3*NPeaks+1))
        
            yconv = np.convolve(y[j,:],np.ones(minsep))[int(np.floor(minsep/2)):-int((minsep-np.floor(minsep/2)))]/minsep
        
            # find index position and amplitude of peaks 
            peakarg = np.array(np.where(((np.diff(yconv) < 0) * np.roll((np.diff(yconv) > 0), 1)) == True))[0]
            peakamp = np.take(y[j,:], peakarg)
        
            # nearest separation of each peak with neigboring peaks
            minpeakargdiff = np.array([np.min(abs(peakarg[k] - np.roll(peakarg,-k)[1:np.size(peakarg)])) for k in np.arange(np.size(peakarg))])

            # minor peaks conditions : 
            #    cond01 - right side higher than current and separation smaller than min separation
            #    cond02 - left side higher than current and separation smaller than min separation
            #    cond03 - smaller than a certain fraction of the highest peak amplitude
            cond01 = (np.diff(peakamp) > 0)*(abs(np.diff(peakarg)) < minsep) 
            cond01 = np.append(cond01,[False])
            cond02 = (np.diff(np.roll(peakamp,1)) < 0)*(abs(np.diff(np.roll(peakarg,1))) < minsep) 
            cond02 = np.append(cond02,[((peakamp[-1]-peakamp[-2]) < 0)*(abs(peakarg[-1]-peakarg[-2]) < minsep)])
            cond02[0] = False
            cond03 = peakamp < 0.2*np.max(peakamp)
        
            # delete minor peaks from list
            minorpeaksind = np.array(np.where((cond01 + cond02 + cond03) == True))[0]
            peakarg = np.delete(peakarg, minorpeaksind)
            peakamp = np.delete(peakamp, minorpeaksind)
            minpeakargdiff = np.delete(minpeakargdiff, minorpeaksind)
        
            # sort peaks according to amplitude and separation
            sortind = np.lexsort((minpeakargdiff,peakamp))[::-1]
            peaksort = np.array([(peakarg[k],peakamp[k]) for k in sortind])
        
            # set minimum peak number for fitting
            Nfitpeak = np.min([NPeaks,np.size(peaksort,0)])
            if Nfitpeak < 1: Nfitpeak = 1
            
            repeat_count = 0
            repeat_flag = 1
            
            while(repeat_flag == 1 and repeat_count <= 5):
            
                try:
                    # Initial guess for the parameters
                    initial_guess = np.zeros(int(Nfitpeak*3 + 1))
            
                    # Bound
                    bl = np.zeros(int(Nfitpeak*3 + 1))
                    bu = np.zeros(int(Nfitpeak*3 + 1))
            
                    for k in range(Nfitpeak):
                        initial_guess[int(k*3)] = abs(peaksort[k,0])
                        initial_guess[int(k*3 + 1)] = 5
                        initial_guess[int(k*3 + 2)] = abs(peaksort[k,1])
                        bl[int(k*3)], bu[int(k*3)] = -np.inf, np.inf
                        bl[int(k*3 + 1)], bu[int(k*3 + 1)] = -np.inf, np.inf
                        bl[int(k*3 + 2)], bu[int(k*3 + 2)] = -np.inf, np.inf
                    
                    if(repeat_count >= 3):
                        initial_guess = [abs(np.argmax(y[j,:])), 5, abs(np.max(y[j,:])), 0]
                        
                    if(repeat_count >= 5):
                        initial_guess = [int(HalfNSpecSam/2), 5, 1000, 0]
                    
                    baseline = (np.mean(y[0,1:10]) + np.mean(y[1,-10:-1]))/2    
                    initial_guess[-1] = baseline
                    bl[-1], bu[-1] = -np.inf, np.inf
            
                    params, covar = curve_fit(lorentzian_multipeaks, x, y[j,:], p0=initial_guess, bounds=(bl, bu))
                
                except:
                    params = np.zeros(int(3*NPeaks+1))
                
                # sorting the fitted parameters accroding to the amplitude
                amplist = np.zeros(Nfitpeak)
                ampsortind = np.zeros(Nfitpeak)
                for k in range(Nfitpeak):
                    amplist[k] = params[int(k*3+2)]
                ampsortind = np.lexsort((amplist,amplist))[::-1]
                
                for k in range(Nfitpeak):
                    paramssort[int(k*3):int(k*3+3)] = params[int(ampsortind[k]*3):int(ampsortind[k]*3+3)]
                paramssort[-1] = params[-1]
                
                if(paramssort[0] <= 0 or paramssort[1] <= 0 or paramssort[2] <= 0):
                    paramssort = np.zeros(int(3*NPeaks+1))
                    Nfitpeak = 1
                    repeat_flag = 1
                else:
                    repeat_flag = 0
                    
                repeat_count += 1
            
            # loop over each peaks for fitted spectrum and frequency calibrated parameters
            paramfit = np.zeros(int(3*Nfitpeak+1))
            if (j == 0):
                for k in range(Nfitpeak):
                    
                    # the fitted spectrum
                    paramfit[int(k*3)] = paramssort[int(k*3)]
                    paramfit[int(k*3+1)] = paramssort[int(k*3+1)]
                    paramfit[int(k*3+2)] = paramssort[int(k*3+2)]                    
                    
                    # calibrated parameters
                    paramcal[int(k*3)] = fcalpar[j,0]*paramssort[int(k*3)] + fcalpar[j,1]
                    paramcal[int(k*3+1)] = abs(fcalpar[j,0])*paramssort[int(k*3+1)]
                    paramcal[int(k*3+2)] = paramssort[int(k*3+2)]
                    
                iyfitl[i,:] = lorentzian_multipeaks(x, *paramfit)
                iParamsl[i,:] = paramcal
                xfreql[i,:] = fcalpar[j,0]*x + fcalpar[j,1]
                
            else:
                for k in range(Nfitpeak):
                    
                    # the fitted spectrum
                    paramfit[int(k*3)] = paramssort[int(k*3)]
                    paramfit[int(k*3+1)] = paramssort[int(k*3+1)]
                    paramfit[int(k*3+2)] = paramssort[int(k*3+2)]
                    
                    # calibrated parameters
                    paramcal[int(k*3)] = fcalpar[j,0]*(paramssort[int(k*3)] + HalfNSpecSam) + fcalpar[j,1]
                    paramcal[int(k*3+1)] = abs(fcalpar[j,0])*paramssort[int(k*3+1)]
                    paramcal[int(k*3+2)] = paramssort[int(k*3+2)]

                iyfitr[i,:] = lorentzian_multipeaks(x, *paramfit)
                iParamsr[i,:] = paramcal
                xfreqr[i,:] = fcalpar[j,0]*(x + HalfNSpecSam) + fcalpar[j,1]
            
            if(repeat_count > 5):
                Nfitpeak = 0
            else:
                Nfitpklist[i,j] = Nfitpeak
        
        print(" " * 100, end="\r")
        print("." * int(i/NSpec*100) + "_" * (100-int(i/NSpec*100)), end='\r')
    
    return iParamsl, iParamsr, iyfitl, iyfitr, xfreql, xfreqr, Nfitpklist

def LorentzianCalibCF(spectra):
    
    NSpec=np.size(spectra,0)
    iParams=np.zeros([NSpec,7]) 
    x=np.arange(np.size(spectra,1))
    
    print("\n\nCalibration Data Lorentzian Curve Fitting:")
    
    for i in range(NSpec): 

        y=spectra[i,:]   
        
        # Perform curve fitting
        # Extract the fitted parameters > x01_fit, gamma1_fit, A1_fit, x02_fit, gamma2_fit, A2_fit, c0
        
        params=np.zeros(7)
                
        try:
        # Initial guess for the parameters
            initial_guess = [np.argmax(y[0:int(np.size(y)/2)]), 5, np.max(y[0:int(np.size(y)/2)]), 
                             np.argmax(y[int(np.size(y)/2):-1])+int(np.size(y)/2), 5, np.max(y[int(np.size(y)/2):-1]), 
                             (np.mean(y[1:10])+np.mean(y[-10:-1]))/2]  
            params, covar = curve_fit(lorentzian_double, x, y, p0=initial_guess)
                
        except:
            params=np.zeros(7)  
            
        iParams[i,:]=params
            
        print(" " * 100, end="\r")
        print("." * int(i/NSpec*100) + "_" * (100-int(i/NSpec*100)), end='\r')
        
    return iParams

def LinearCalibrationCF(calibration, iCalibParams):
    
    NLine = int(np.size(calibration, 0))
    NCalib = int(np.size(calibration, 1))
    
    CalibParams=np.zeros([NLine,4])
    
    print("\nLinear Calibration Curve Fitting:")
    for i in range(NLine):
        
        # Calibration frequencies as in the unit of index
        iCalib_01=iCalibParams[int(i*NCalib):int((i+1)*NCalib),0]
        iCalib_02=iCalibParams[int(i*NCalib):int((i+1)*NCalib),3]
        
        # Linear fitting of calibration frequencies
        x=iCalib_01
        y=calibration[i,:]
        m_guess=(y[-1]-y[0])/(x[-1]-x[0])
        c_guess=y[0]-m_guess*x[0]
        
        try:
        # Initial guess for the parameters
            initial_guess = [m_guess, c_guess]
            params_01, params_covariance_01 = curve_fit(linear, x, y, p0=initial_guess)
            
        except:
            params_01=np.zeros(2)          
    
        x=iCalib_02
        y=calibration[i,:]
        m_guess=(y[-1]-y[0])/(x[-1]-x[0])
        c_guess=y[0]-m_guess*x[0]
        
        try:
        # Initial guess for the parameters
            initial_guess = [m_guess, c_guess]
            params_02, params_covariance_02 = curve_fit(linear, x, y, p0=initial_guess)
            
        except:
            params_02=np.zeros(2)     
    
        CalibParams[i,:]=np.concatenate((params_01,params_02))

        print(" " * 100, end="\r")
        print("." * int(i/NLine*100) + "_" * (100-int(i/NLine*100)), end='\r')
    
    return CalibParams

def ContourMapData(DataList, iMotorcoods):
    
    NData=np.size(DataList)
    
    NXstep=int(np.max(iMotorcoods[:,0])+1)
    NYstep=int(np.max(iMotorcoods[:,1])+1)
    NZstep=int(np.max(iMotorcoods[:,2])+1)
    
    ContourData=np.zeros([NXstep,NYstep,NZstep])
    for i in range(NData):
        ContourData[int(iMotorcoods[i,0]),int(iMotorcoods[i,1]),int(iMotorcoods[i,2])]=DataList[i]
    
    return ContourData

def iSpectrumImagePosition(ispec, MotorSteps, MDir):
    
    if(ispec < 0):
        ispec = 0
    elif(ispec >= MotorSteps[0]*MotorSteps[1]*MotorSteps[2]):
        ispec = MotorSteps[0]*MotorSteps[1]*MotorSteps[2] - 1

    iplane = np.remainder(ispec, MotorSteps[0]*MotorSteps[1])        
    iz = np.floor(ispec/(MotorSteps[0]*MotorSteps[1]))
    iy = np.floor(iplane/MotorSteps[0])
    ix = np.remainder(iplane, MotorSteps[0])
    
    if(MDir[0] == 0):
        
        if(MDir[1] == 0):
            ix = ix + 1
        else:
            ix = ix - 1
            
        if(ix < 0):
            ix = MotorSteps[0] + ix
        elif(ix >= MotorSteps[0]):
            ix = ix - MotorSteps[0]

    else:
        
        if(MDir[1] == 0):
            iy = iy + 1
        else:
            iy = iy - 1
            
        if(iy < 0):
            iy = MotorSteps[1] + iy
        elif(iy >= MotorSteps[1]):
            iy = iy - MotorSteps[1]        
            
    ispec_update = iz*(MotorSteps[0]*MotorSteps[1]) + iy*MotorSteps[1] + ix

    return int(ispec_update)