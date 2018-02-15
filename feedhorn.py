# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 00:48:49 2016

@author: Matthew Berkeley
"""
import os
import numpy as np
import math
import cmath
import scipy.constants as const

class Feedhorn:
    
    def __init__(self, a, l, w_a_frac=0.76):
        '''Initialize dimensions of feedhorn
        
        Attributes
        ----------
        
        .w_a_frac : float
            Beam radius to aperture fraction 
            (=0.76 for smooth-walled circular feedhorn) \n
        .a : float
            Feedhorn aperture radius \n
        .l : float
            Feedhorn length
        '''
        
        self.w_a_frac = w_a_frac
        self.a = a
        self.l = l
        
        
        
class BeamMap():
    '''Class object to be used for feedhorn beam mapping experiments.
    
    Caution: Only use data taken with the same Network Analyzer settings for 
    any particular plane (E or H). 
    This is because the frequency information is only saved from one input file 
    and assumed to be the same for the rest. 
    
    Attributes
    ----------
    
    x0 : float
        Distance between rotating feedhorn aperture and center of rotation
    y0 : float
        Distance between stationary feedhorn aperture and center of rotation
    zp : float
        Zero angle reference point (i.e. angle reading on rotation scale when 
        feedhorns are aligned)
    fhorn_stat : object
        Feedhorn object representing stationary feedhorn
    fhorn_rot : object
        Feedhorn object representing rotating feedhorn
    Ecopol_angles : 1darray
        Array of angles to be plotted in the E plane.
    Ecopol_freqs : 1darray
        Array of frequencies at which E plane data was taken.
    Ecopol_field : ndarray
        Array of arrays containing E field copolarization information (real and 
        imaginary parts) for each frequency
    Ecopol_power : ndarray
        Array of arrays containing E power copolarization information (in dB) 
        for each frequency
    Ecopol_angles_adj : ndarray
        Array of arrays containing modified E plane angle information for each 
        frequency
    Hcopol_angles : 1darray
        Array of angles to be plotted in the H plane.
    Hcopol_freqs : 1darray
        Array of frequencies at which H plane data was taken.
    Hcopol_field : ndarray
        Array of arrays containing H field copolarization information (real and 
        imaginary parts) for each frequency
    Hcopol_power : ndarray
        Array of arrays containing H power copolarization information (in dB) 
        for each frequency
    Dcopol_field : ndarray
        Array of arrays containing D field copolarization information (real and 
        imaginary parts) for each frequency
    Dcopol_power : ndarray
        Array of arrays containing D power copolarization information (in dB) 
        for each frequency
    Dcross_field : ndarray
        Array of arrays containing D field x-polarization information (real and 
        imaginary parts) for each frequency
    Dcross_power : ndarray
        Array of arrays containing D power x-polarization information (in dB) 
        for each frequency
    '''
    
    def __init__(self, x0, y0, zp, fhorn_stat, fhorn_rot):
        '''Initialize beam mapping configuration
        
        Parameters
        ----------
        
        x0 : float
            Distance between rotating feedhorn aperture and center of rotation
        y0 : float
            Distance between stationary feedhorn aperture and center of 
            rotation
        zp : float
            Zero angle reference point (i.e. angle reading on rotation scale 
            when feedhorns are aligned)
        fhorn_stat : object
            Feedhorn object representing stationary feedhorn
        fhorn_rot : object
            Feedhorn object representing rotating feedhorn
        
        '''
        
        self.x0 = x0
        self.y0 = y0
        self.zp = zp
        self.fhorn_stat = fhorn_stat
        self.fhorn_rot = fhorn_rot
        
    def load_Eplane(self, tag='Ecopol', path='./'):
        '''Load E plane data.
        
        Parameters
        ----------
        
        tag : str
            Expression in file name to denote E plane data. 
            Default is 'Ecopol'.
        path : str
            Path to data files.
        '''
        Ecopol_files = []
        Ecopol_field_perang = []
        Ecopol_power_perang = []
        self.Ecopol_angles = []
        # The files are read in assuming each data file contains the specified 
        # tag and a three character angle before the '.s2p' extension.
        for arg in os.listdir(path):
            if tag in arg and 's2p' in arg:
                Ecopol_files.append(path+arg)
                self.Ecopol_angles.append(float(arg[-7:-4])-self.zp)
        # Sort the list of files in order of increasing angle so they are 
        # processed correctly. This is necessary because the files are read in 
        # alphabetically, which causes files with a '-' sign in the angle to be 
        # out of order.        
        Ecopol_files, self.Ecopol_angles = np.array(
            [(x,y) for (y,x) in sorted(zip(self.Ecopol_angles, Ecopol_files), 
            key=lambda pair: pair[0])]
            ).T 
            
        # Given the way data was taken, most angles were negative. This just 
        # flips the sign so the angles plotted are mostly positive.     
        self.Ecopol_angles = np.array([float(x) for x in self.Ecopol_angles]) 
        
        for file in Ecopol_files:
            with open(file,'r') as f:
                (f_ec,s11r_ec,s11i_ec,s12r_ec,s12i_ec,
                 s21r_ec,s21i_ec,s22r_ec,s22i_ec) = np.loadtxt(
                     f, skiprows=9, unpack=True
                     )
                Ecopol_field_perang.append(
                    [(s21r_ec[i] + 1.j*s21i_ec[i]) for i in xrange(len(f_ec))]
                    )
                Ecopol_power_perang.append(
                    [20*math.log10(math.sqrt(s21r_ec[i]**2+s21i_ec[i]**2)) 
                    for i in xrange(len(f_ec))]
                    )
                self.Ecopol_freqs = f_ec
        self.Ecopol_field = np.array(Ecopol_field_perang).T
        self.Ecopol_power = np.array(Ecopol_power_perang).T
        self.Ecopol_angles_adj = self.correct_angles(
            self.Ecopol_angles, self.Ecopol_freqs, phase_center_2=0.00812
            )
        return
        

    def load_Hplane(self, tag='Hcopol', path='./'):
        '''Load H plane data.
        
        Parameters
        ----------
        
        tag : str
            Expression in file name to denote H plane data. 
            Default is 'Hcopol'.
        path : str
            Path to data files.
        '''
        Hcopol_files = []
        Hcopol_field_perang = []
        Hcopol_power_perang = []
        self.Hcopol_angles = []
        # The files are read in assuming each data file contains the specified 
        # tag and a three character angle before the '.s2p' extension.
        for arg in os.listdir(path):
            if tag in arg and 's2p' in arg:
                Hcopol_files.append(path+arg)
                self.Hcopol_angles.append(float(arg[-7:-4])-self.zp)
        # Sort the list of files in order of increasing angle so they are 
        # processed correctly. This is necessary because the files are read in 
        # alphabetically, which causes files with a '-' sign in the angle to be 
        # out of order. 
        Hcopol_files, self.Hcopol_angles = np.array(
            [(x,y) for (y,x) in sorted(zip(self.Hcopol_angles, Hcopol_files), 
            key=lambda pair: pair[0])]
            ).T
        self.Hcopol_angles = np.array([float(x) for x in self.Hcopol_angles])
        for file in Hcopol_files:
            with open(file,'r') as g:
                (f_hc,s11r_hc,s11i_hc,s12r_hc,s12i_hc,
                 s21r_hc,s21i_hc,s22r_hc,s22i_hc) = np.loadtxt(
                     g, skiprows=9, unpack=True
                     )
                Hcopol_field_perang.append(
                    [(s21r_hc[i] + 1.j*s21i_hc[i]) for i in xrange(len(f_hc))]
                    )
                Hcopol_power_perang.append(
                    [20*math.log10(math.sqrt(s21r_hc[i]**2+s21i_hc[i]**2)) 
                    for i in xrange(len(f_hc))]
                    )
                self.Hcopol_freqs = f_hc
        self.Hcopol_field = np.array(Hcopol_field_perang).T
        self.Hcopol_power = np.array(Hcopol_power_perang).T
        self.Hcopol_angles_adj = self.correct_angles(
            self.Hcopol_angles, self.Hcopol_freqs, phase_center_2=0.00967
            )
        return
        
    def load_Dplanes(self, Etag='Ecopol', Htag='Hcopol', path='./'):
        '''Derive D plane data from input E and H plane data.
        
        Parameters
        ----------
        
        Etag : str
            Expression in file name to denote E plane copolarization data. 
            Default is 'Ecopol'.
        Htag : str
            Expression in file name to denote H plane copolarization data. 
            Default is 'Hcopol'.
        path : str
            Path to data files.
        '''
        if len(self.Ecopol_power) == 0:
            self.load_Eplane(tag=Etag, path=path)
        if len(self.Hcopol_power) == 0:
            self.load_Hplane(tag=Htag, path=path)
        Hfieldr_adj = [
            np.interp(
                self.Ecopol_angles_adj,self.Hcopol_angles_adj,
                self.Hcopol_field[i].real
                ) for i in xrange(len(self.Hcopol_field))
            ]
        Hfieldi_adj = [
            np.interp(
                self.Ecopol_angles_adj,self.Hcopol_angles_adj,
                self.Hcopol_field[i].imag
                ) for i in xrange(len(self.Hcopol_field))
            ]
        Hfield_adj = [
            [Hfieldr_adj[p][q] + 1.j*Hfieldi_adj[p][q] 
            for q in xrange(len(Hfieldr_adj[p]))] 
            for p in xrange(len(Hfieldr_adj))
            ]
        Hpower_adj = [
            np.interp(
                self.Ecopol_angles_adj,self.Hcopol_angles_adj,
                self.Hcopol_power[i]
                ) for i in xrange(len(self.Hcopol_power))
                ]
        self.Hcopol_field = np.array(Hfield_adj)
        self.Hcopol_power = np.array(Hpower_adj)
        maxvalsE = [
            max(
                [abs(self.Ecopol_field[i][j]) 
                for j in xrange(len(self.Ecopol_field[i]))]
                ) for i in xrange(len(self.Ecopol_field))
            ]
        maxvalsH = [
            max(
                [abs(self.Hcopol_field[i][j]) 
                for j in xrange(len(self.Hcopol_field[i]))]
                ) for i in xrange(len(self.Hcopol_field))
            ]
        self.Dcopol_field = [
                [
                0.5*(abs(self.Ecopol_field[p][q])/maxvalsE[p] 
                + abs(self.Hcopol_field[p][q])/maxvalsH[p]) 
                for q in xrange(len(self.Ecopol_field[p]))
                ] for p in xrange(len(self.Ecopol_field))
            ]
        self.Dcross_field = [
                [
                0.5*(abs(self.Ecopol_field[p][q])/maxvalsE[p] 
                - abs(self.Hcopol_field[p][q])/maxvalsH[p]) 
                for q in xrange(len(self.Ecopol_field[p]))
                ] for p in xrange(len(self.Ecopol_field))
            ]
        self.Dcopol_power = [
                [
                20*math.log10(math.sqrt(self.Dcopol_field[p][q].real**2 
                + self.Dcopol_field[p][q].imag**2)) 
                for q in xrange(len(self.Dcopol_field[p]))
                ] for p in xrange(len(self.Dcopol_field))
            ]
        self.Dcross_power = []
        for i in xrange(len(self.Dcross_field)):
            for j in xrange(len(self.Dcross_field[i])):
                if self.Dcross_field[i][j] != 0.:
                    self.Dcross_power.append(
                        20*math.log10(math.sqrt(
                            self.Dcross_field[i][j].real**2 
                            + self.Dcross_field[i][j].imag**2)
                            )
                        )
                else:
                    self.Dcross_power.append(-100.0)
        self.Dcross_power = np.array(self.Dcross_power).reshape(
            np.array(self.Ecopol_power).shape
            )
        return        
        
    def calc_phase_center(self, freq_list):
        '''Calculate the phase center for the feedhorn at each frequency.
        
        Parameters
        ----------
        
        freq_list : 1darray
            List of frequencies at which data was taken. This should be an 
            attribute of the Feedhorn class, either 'Ecopol_freqs' or 
            'Hcopol_freqs'.
        
        Returns
        -------
        
        offset1 : float
            Distance from feedhorn aperture to phase center for stationary 
            feedhorn, in the same units as Feedhorn attributes 'a' and 'l'.
        offset2 : float
            Distance from feedhorn aperture to phase center for rotating 
            feedhorn, in the same units as Feedhorn attributes 'a' and 'l'.
        '''
        phase_center_1 = []
        phase_center_2 = []
        # Calculate beam radius
        w1 = self.fhorn_stat.w_a_frac*self.fhorn_stat.a
        w2 = self.fhorn_rot.w_a_frac*self.fhorn_rot.a
        # Calculate approximate feedhorn slant length as a proxy for 
        # radius of curvature.
        R1 = math.sqrt((self.fhorn_stat.a/2.)**2+self.fhorn_stat.l**2)
        R2 = math.sqrt((self.fhorn_rot.a/2.)**2+self.fhorn_rot.l**2)    
        # The equations below are Equations 7.30a, 7.30b and 7.45 from 
        # Goldsmith's 'Quasioptical Systems'.
        for j in xrange(len(freq_list)):
            freq = freq_list[j]
            wavelength = const.c/freq
            w0_1 = w1/((1+(math.pi*w1**2/(wavelength*R1))**2)**0.5)
            w0_2 = w2/((1+(math.pi*w2**2/(wavelength*R2))**2)**0.5)
            z1 = R1/(1+(wavelength*R1/(math.pi*w1**2))**2)
            z2 = R2/(1+(wavelength*R2/(math.pi*w2**2))**2)
            # The 'phase_center' is the distance from beam waist location to 
            # the phase_center; the 'offset' is the distance of the 
            # phase_center from the aperture.
            phase_center_1.append(-(math.pi*w0_1**2/wavelength)**2/z1)
            phase_center_2.append(-(math.pi*w0_2**2/wavelength)**2/z2)
            offset1 = [
                z1 - phase_center_1[i] 
                for i in xrange(len(phase_center_1))
                ]
            offset2 = [
                z2 - phase_center_2[i] 
                for i in xrange(len(phase_center_2))
                ]
        return offset1, offset2
        
        
    def correct_angles(self, angle_list, freq_list, 
                       phase_center_1=None, phase_center_2=None):
        '''Adjust measured angles to account for the offset of the phase 
        center from the aperture.
        
        Parameters
        ----------
        
        angle_list : 1darray
            List of angles measured. Should be stored as a Feedhorn class 
            attribute as either 'Ecopol_angles' or 'Hcopol_angles'.
        freq_list : 1darray
            List of frequencies measured. Should be stored as a Feedhorn class 
            attribute as either 'Ecopol_freqs' or 'Hcopol_freqs'.
        phase_center_1 : (optional) float or None
            Phase center offset for stationary feedhorn. If not given, it is 
            calculated.
        phase_center_2 : (optional) float or None
            Phase center offset for rotating feedhorn. If not given, it is 
            calculated.
            
        Returns
        -------
        
        corrected_angles : ndarray
            Array of arrays, containing an updated set of angles for each 
            frequency measured.
        '''
        if self.fhorn_stat.a == None or self.fhorn_rot.a == None:
            return angle_list
        if phase_center_1 == None or phase_center_2 == None:
            phase_center_1, phase_center_2 = self.calc_phase_center(freq_list)
            x = self.x0 + abs(np.mean(phase_center_1))
            y = abs(np.mean(phase_center_2))
        else:
            x = phase_center_1
            y = phase_center_2
            
        # This formula assumes the phase center of the rotating feedhorn is 
        # behind the center of rotation.
        corrected_angles = [
            math.degrees(
                math.asin(
                    x*math.sin(math.radians(angle_list[i]))
                    )
                / math.sqrt(
                    x**2 + y**2 + 2*x*y*math.cos(math.radians(angle_list[i]))
                    )
                ) for i in xrange(len(angle_list))
            ]
        return corrected_angles
        
    def find_freq_index(self, freq, freq_list_E, freq_list_H):
        '''Select index within frequency list that is closest to a specified 
        frequency.
        
        Parameters
        ----------
        
        freq : float
            Specified frequency in GHz.
        freq_list_E : 1darray
            Array containing frequencies measured for E plane data.
        freq_list_H : 1darray
            Array containing frequencies measured for H plane data.
        
        Returns
        -------
        
        E_ind : int
            Index corresponding to the nearest frequency measured. If E plane 
            data not entered, returns None.
        H_ind : int
            Index corresponding to the nearest frequency measured. If H plane 
            data not entered, returns None.
        '''
        try:
            E_ind = np.where(
                freq_list_E == min(
                    freq_list_E, key=lambda x: abs(freq*10**9 - x)
                    )
                )[0][0]
        except:
            E_ind = None
        try:
            H_ind = np.where(
                freq_list_H == min(
                    freq_list_H, key=lambda x: abs(freq*10**9 - x)
                    )
                )[0][0]
        except:
            H_ind = None
        return E_ind, H_ind

class Model:
    '''Model class containing three model options: KP, LZ and BB, named after 
    Kongpop, Lingzhen and Berhanu Bulcha respectively.
    
    KP and BB used HFSS to create their models. LZ used a mode matching method.
    The data can be found in the appropriate directory.
    
    Also included is an option to load data from the CMI website for a 
    WR10 RCHO10R feedhorn.
    
    Methods
    -------
    
    load_KP_model : 
    
    load_LZ_model :
    
    load_BB_model :
    
    load_CMI_model :
    
    find_freq_index :
    
    
    Attributes
    ----------
    
    Ecopol_power_KP80 :
    
    Hcopol_power_KP80 :

    Ecopol_power_KP90 :

    Hcopol_power_KP90 :

    Ecopol_power_KP100 :

    Hcopol_power_KP100 :

    angles_KP :

    Ecopol_field_LZ :

    Hcopol_field_LZ :

    Dcopol_field_LZ :

    Dcross_field_LZ :

    Ecopol_power_LZ :

    Hcopol_power_LZ :

    Dcopol_power_LZ :

    Dcross_power_LZ :

    angles_LZ :

    freqs_LZ :

    angles_BB :

    Ecopol_power_BB925 :

    Hcopol_power_BB925 :
    
    ang_E_CMI : 
    
    power_E_CMI :
    
    ang_H_CMI : 
    
    power_H_CMI :

    '''
    
    def __init__(self):
        return
        
    def load_KP_model(self, filepath='./feedhorn_model.csv'):
        '''KPs model gives the realized gain (no field data) for three 
        frequencies: 80, 90 and 100 GHz.
        
        Parameters
        ----------
        
        filepath : str
            Path to model datafile.        
        '''
        with open(filepath, 'r') as f:
            (angle, ec_80_0, ec_80_90, ec_90_0, ec_90_90, 
                 ec_100_0, ec_100_90, ex_80_0, ex_80_90, ex_90_0, ex_90_90, 
                 ex_100_0, ex_100_90) = np.loadtxt(
                     f, skiprows=1, delimiter=',', unpack=True
                     )     
            model_angs = ([0-ang for ang in angle[60:0:-1].tolist()] 
                + angle[:60].tolist())
            model_ec_80 = (ec_80_0[60:0:-1].tolist() 
                + ec_80_0[:60].tolist())
            model_hc_80 = (ec_80_90[60:0:-1].tolist() 
                + ec_80_90[:60].tolist())
            model_ec_90 = (ec_90_0[60:0:-1].tolist() 
                + ec_90_0[:60].tolist())
            model_hc_90 = (ec_90_90[60:0:-1].tolist() 
                + ec_90_90[:60].tolist())
            model_ec_100 = (ec_100_0[60:0:-1].tolist() 
                + ec_100_0[:60].tolist())
            model_hc_100 = (ec_100_90[60:0:-1].tolist() 
                + ec_100_90[:60].tolist())
            self.Ecopol_power_KP80 = np.array(
                [10*math.log10(model_ec_80[i]) 
                for i in xrange(len(model_ec_80))]
                )
            self.Hcopol_power_KP80 = np.array(
                [10*math.log10(model_hc_80[i]) 
                for i in xrange(len(model_hc_80))]
                )
            self.Ecopol_power_KP90 = np.array(
                [10*math.log10(model_ec_90[i]) 
                for i in xrange(len(model_ec_90))]
                )
            self.Hcopol_power_KP90 = np.array(
                [10*math.log10(model_hc_90[i]) 
                for i in xrange(len(model_hc_90))]
                )
            self.Ecopol_power_KP100 = np.array(
                [10*math.log10(model_ec_100[i]) 
                for i in xrange(len(model_ec_100))]
                )
            self.Hcopol_power_KP100 = np.array(
                [10*math.log10(model_hc_100[i]) 
                for i in xrange(len(model_hc_100))]
                )
            self.angles_KP = model_angs
            
    def load_LZ_model(self, path='.'):
        '''LZs model gives multi-mode data stored in '*beam.out' files.
        
        Parameters
        ----------
        
        path : str
            Path to model datafile.        
        '''
        self.Ecopol_field_LZ = []
        self.Hcopol_field_LZ = []
        self.Dcopol_field_LZ = []
        self.Dcross_field_LZ = []
        self.Ecopol_power_LZ = []
        self.Hcopol_power_LZ = []
        self.Dcopol_power_LZ = []
        self.Dcross_power_LZ = []
        
        model_files = []
        model_freqs = []
        for arg in os.listdir(path):
            if 'beam.out' in arg:
                model_files.append(path+arg)
                # The frequencies are divided by 1.96 because the input files
                # should be from the CLASS HF model.
                model_freqs.append((1./1.96)*float(arg[3:8]))                
        
        for datafile in model_files:            
            with open(datafile, 'r') as f:
                angle,Emag,Ephase,Hmag,Hphase=np.loadtxt(f,unpack=True)
                model_angs = ([0-ang for ang in angle[1000:0:-1].tolist()] 
                    + angle.tolist())
                self.angles_LZ = model_angs
                Eplane = [
                    Emag[i]*cmath.exp(cmath.sqrt(-1)*Ephase[i]*(math.pi)/180) 
                    for i in xrange(len(angle))
                    ]
                full_Eplane = Eplane[1000:0:-1] + Eplane
                Hplane = [
                    Hmag[i]*cmath.exp(cmath.sqrt(-1)*Hphase[i]*(math.pi)/180) 
                    for i in xrange(len(angle))
                    ]
                full_Hplane = Hplane[1000:0:-1] + Hplane
                self.Ecopol_field_LZ.append(full_Eplane)
                self.Hcopol_field_LZ.append(full_Hplane)
                Dcross = [
                    0.5*(full_Eplane[j]-full_Hplane[j]) 
                    for j in xrange(len(full_Eplane))
                    ]
                Dcopol = [
                    0.5*(full_Eplane[j]+full_Hplane[j]) 
                    for j in xrange(len(full_Eplane))
                    ]
                self.Dcross_field_LZ.append(Dcross)
                self.Dcopol_field_LZ.append(Dcopol)
                for i in xrange(len(Dcross)):
                    if Dcross[i] != 0.:
                        self.Dcross_power_LZ.append(
                            20*math.log10(math.sqrt(
                                Dcross[i].real**2 + Dcross[i].imag**2
                                ))
                            )
                    else:
                        self.Dcross_power_LZ.append(-100.0)
                for i in xrange(len(Dcopol)):
                    if Dcopol[i] != 0:
                        self.Dcopol_power_LZ.append(
                            20*math.log10(math.sqrt(
                                Dcopol[i].real**2 + Dcopol[i].imag**2
                                ))
                            )
                    else:
                        self.Dcopol_power_LZ.append(-100.0)
                for i in xrange(len(full_Eplane)):
                    if full_Eplane[i] != 0.:
                        self.Ecopol_power_LZ.append(
                            20*math.log10(math.sqrt(
                                full_Eplane[i].real**2 + full_Eplane[i].imag**2
                                ))
                            )
                    else:
                        self.Ecopol_power_LZ.append(-100.0)
                
                for i in xrange(len(full_Hplane)):
                    if full_Hplane[i] != 0.:
                        self.Hcopol_power_LZ.append(
                            20*math.log10(math.sqrt(
                                full_Hplane[i].real**2 + full_Hplane[i].imag**2
                                ))
                            )
                    else:
                        self.Hcopol_power_LZ.append(-100.0)
        
        self.Ecopol_power_LZ = np.array(self.Ecopol_power_LZ).reshape(
            (len(model_files), len(full_Eplane))
            )
        self.Hcopol_power_LZ = np.array(self.Hcopol_power_LZ).reshape(
            (len(model_files), len(full_Hplane))
            )
        self.Dcross_power_LZ = np.array(self.Dcross_power_LZ).reshape(
            np.array(self.Ecopol_power_LZ).shape
            )
        self.Dcopol_power_LZ = np.array(self.Dcopol_power_LZ).reshape(
            np.array(self.Ecopol_power_LZ).shape
            )
        self.freqs_LZ = model_freqs
        
    def load_BB_model(self, filepath='model_data_0622.csv'):
        '''BBs model gives the expected beam plot at 92.5 GHz.
        
        Parameters
        ----------
        
        filepath : str
            Path to model datafile.        
        '''
        
        with open(filepath, 'rb') as f:
            ang_E, power_E, power_H = np.genfromtxt(
                f, dtype=float, delimiter=',', skip_header=1, unpack=True
                )
        self.angles_BB = ang_E
        self.Ecopol_power_BB925 = power_E - max(power_E)
        self.Hcopol_power_BB925 = power_H - max(power_H)
        
    def load_CMI_model(self, filepath='.'):
        '''CMI model data extracted from an image of a plot on their website 
        using Dexter.
        
        Parameters
        ----------
        
        filepath : str
            Path to model datafiles, named 'beam_pattern_Eplane.1' and 
            'beam_pattern_Hplane.1'        
        '''
        with open('beam_pattern_Eplane.1', 'rb') as f:
            self.ang_E_CMI, self.power_E_CMI = np.genfromtxt(
                f, dtype=float, skip_header=1, unpack=True
                )
        with open('beam_pattern_Hplane.1', 'rb') as f:
            self.ang_H_CMI, self.power_H_CMI = np.genfromtxt(
                f, dtype=float, skip_header=1, unpack=True
                )
        
    def find_freq_index(self, freq, freq_list):
        '''Select index within frequency list that is closest to a specified 
        frequency.
        
        Parameters
        ----------
        
        freq : float
            Specified frequency in GHz.
        freq_list : 1darray
            Array containing frequencies measured.
        
        Returns
        -------
        
        E_ind : int
            Index corresponding to the nearest frequency measured.
        H_ind : int
            Index corresponding to the nearest frequency measured.
        '''
        E_ind = H_ind = np.where(
            np.array(freq_list) == min(
                freq_list, key=lambda x: abs(freq - x)
                )
            )[0][0]
        return E_ind, H_ind