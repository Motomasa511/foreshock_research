c     interpolate.f
c     calculate velocity and reliability at any point
c
      character*80 param,outfile
      dimension rlon(310),rlat(290),dep(50)
      dimension vp(310,290,50),vs(310,290,50),
     $     rp(310,290,50),rs(310,290,50)
c
c  LAT.lst  : list of latitude
c  LON.lst  : list of longitude
c  DEP.lst  : list of depth
c  VEL.dat  : list of velocity data
c  param    : the number of points and list of coodinates of points
c
      parameter (rlatint=0.1000, rlonint=0.1000)
      parameter (numlon=301, numlat=281, numdep=32)
      parameter (rlonw=120.000, rlats=20.000)
c      write(6,*) "Input parameter file name"
c      read(5,'(a)') param
c      write(6,*) "Input output file name"
c      read(5,'(a)') outfile

      open(11, file="LON.lst", status='old')
      open(12, file="LAT.lst", status='old')
      open(13, file="DEP.lst", status='old')
      open(14, file="VEL.dat", status='old')
      open(15, file="param", status='old')
      open(51, file="outfile", status='unknown')
c
      do 10 i=1, numlon
         read(11,*) rlon(i)
 10   continue
      do 20 j=1, numlat
         read(12,*) rlat(j)
 20   continue
      do 30 k=1, numdep
         read(13,*) dep(k)
 30   continue
      do 40 k=1,numdep
         do 50 i = 1, numlon
            do 60 j = 1, numlat
               read(14,*) rlon1, rlat1, dep1,
     $              vp(i,j,k), vs(i,j,k), rp(i,j,k), rs(i,j,k)
               if(rlon1.ne.rlon(i).or.rlat1.ne.rlat(j).
     $              or.dep1.ne.dep(k)) then
                  write(*,*) "input data error"
                  stop
               endif
 60         continue
 50      continue
 40   continue

      read(15,*) numdata
      do 70 l=1,numdata
         read(15,*) x,y,z
         i0=int((x-rlonw)/rlonint)+1
         i1=i0+1
         j0=int((y-rlats)/rlatint)+1
         j1=j0+1
         do 80 k=1, numdep
            if(dep(k).gt.z) then
               k0=k-1
               k1=k
               goto 90
            endif
 80      continue
 90      continue
         vp000=1./vp(i0,j0,k0)
         vs000=1./vs(i0,j0,k0)
         rp000=rp(i0,j0,k0)
         rs000=rs(i0,j0,k0)
         vp010=1./vp(i0,j1,k0)
         vs010=1./vs(i0,j1,k0)
         rp010=rp(i0,j1,k0)
         rs010=rs(i0,j1,k0)
         vp100=1./vp(i1,j0,k0)
         vs100=1./vs(i1,j0,k0)
         rp100=rp(i1,j0,k0)
         rs100=rs(i1,j0,k0)
         vp110=1./vp(i1,j1,k0)
         vs110=1./vs(i1,j1,k0)
         rp110=rp(i1,j1,k0)
         rs110=rs(i1,j1,k0)
         vp001=1./vp(i0,j0,k1)
         vs001=1./vs(i0,j0,k1)
         rp001=rp(i0,j0,k1)
         rs001=rs(i0,j0,k1)
         vp011=1./vp(i0,j1,k1)
         vs011=1./vs(i0,j1,k1)
         rp011=rp(i0,j1,k1)
         rs011=rs(i0,j1,k1)
         vp101=1./vp(i1,j0,k1)
         vs101=1./vs(i1,j0,k1)
         rp101=rp(i1,j0,k1)
         rs101=rs(i1,j0,k1)
         vp111=1./vp(i1,j1,k1)
         vs111=1./vs(i1,j1,k1)
         rp111=rp(i1,j1,k1)
         rs111=rs(i1,j1,k1)
         x0=rlon(i0)
         x1=rlon(i1)
         y0=rlat(j0)
         y1=rlat(j1)
         z0=dep(k0)
         z1=dep(k1)
         dx=x1-x0
         dy=y1-y0
         dz=z1-z0

         vp0=1./((vp000*(x1-x)*(y1-y)*(z1-z)+vp010*(x1-x)*(y-y0)*(z1-z)+
     $        vp100*(x-x0)*(y1-y)*(z1-z)+vp110*(x-x0)*(y-y0)*(z1-z)+
     $        vp001*(x1-x)*(y1-y)*(z-z0)+vp011*(x1-x)*(y-y0)*(z-z0)+
     $        vp101*(x-x0)*(y1-y)*(z-z0)+vp111*(x-x0)*(y-y0)*(z-z0))
     $        /dx/dy/dz)
         vs0=1./((vs000*(x1-x)*(y1-y)*(z1-z)+vs010*(x1-x)*(y-y0)*(z1-z)+
     $        vs100*(x-x0)*(y1-y)*(z1-z)+vs110*(x-x0)*(y-y0)*(z1-z)+
     $        vs001*(x1-x)*(y1-y)*(z-z0)+vs011*(x1-x)*(y-y0)*(z-z0)+
     $        vs101*(x-x0)*(y1-y)*(z-z0)+vs111*(x-x0)*(y-y0)*(z-z0))
     $        /dx/dy/dz)
         rp0=(rp000*(x1-x)*(y1-y)*(z1-z)+rp010*(x1-x)*(y-y0)*(z1-z)+
     $        rp100*(x-x0)*(y1-y)*(z1-z)+rp110*(x-x0)*(y-y0)*(z1-z)+
     $        rp001*(x1-x)*(y1-y)*(z-z0)+rp011*(x1-x)*(y-y0)*(z-z0)+
     $        rp101*(x-x0)*(y1-y)*(z-z0)+rp111*(x-x0)*(y-y0)*(z-z0))
     $        /dx/dy/dz
         rs0=(rs000*(x1-x)*(y1-y)*(z1-z)+rs010*(x1-x)*(y-y0)*(z1-z)+
     $        rs100*(x-x0)*(y1-y)*(z1-z)+rs110*(x-x0)*(y-y0)*(z1-z)+
     $        rs001*(x1-x)*(y1-y)*(z-z0)+rs011*(x1-x)*(y-y0)*(z-z0)+
     $        rs101*(x-x0)*(y1-y)*(z-z0)+rs111*(x-x0)*(y-y0)*(z-z0))
     $        /dx/dy/dz
         if(rp0.gt.0.8.or.rp0.lt.0.0) vp0=0.1
         if(rs0.gt.0.8.or.rs0.lt.0.0) vs0=0.1
         write(51,500) x,y,z,vp0,vs0,rp0,rs0
 70   continue
 500  format(f7.3,f7.3,f7.2,f7.3,f6.3,f7.3,f7.3)
      stop
      end
