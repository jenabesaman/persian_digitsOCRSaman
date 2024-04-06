{---------------------------------------------------------------------------|
|						In the name of ALLAH								|
|	This is a simple pascal unit for reading *.cdb files in windows OS		|
|	Auther: Hossein Khosravi 												|
|	Primary Email Address: HosseinKhosravi@gmail.com 						|
|	Other Email Addresses: Khosravi@HodaSystem.com, Khosrav@modares.ac.ir	|
|	August 2005, Tehran, Iran
----------------------------------------------------------------------------}
unit ReadCDB;

interface
uses Windows, sysutils;

type
   TbyteMx = array of array of byte;
   TData = array of TbyteMx;//images of all characters in database
   TLabels = array of byte;//corresponding labels
   TImageType = (itBinary, itGray);

//FileName: is the path of CDB file in hard disk.
//Data: Samples from CDB file will be stored in Data
procedure ReadData(FileName: string; var Data: TData; var labels: TLabels; var imgType: TimageType);
procedure ReadRamFile(var Buffer; ByteCount:integer);

var
   CurAddr: Cardinal = 0;
implementation

procedure ReadData(FileName: string; var Data: TData; var labels: TLabels; var imgType: TimageType);
var
   hFile, hMap: Cardinal;
   pBase: Pointer;
   d,m,w,h,x,y,StartByte,counter,WBcount: BYTE;
   yy: WORD;
   TotalRec: DWORD;
   LetterCount: array of DWORD;
   Comments: array[0..255] of Char;
   normal,bWhite: Boolean;
   i: integer;
   ByteCount: WORD;
begin
   hFile:= CreateFile(PansiChar(FileName), GENERIC_READ,
                 FILE_SHARE_READ, NIL, OPEN_EXISTING, 0, 0);
   if(Failed(hFile)) then
   begin
      hFile:= 0;
      MessageBox(HWND(Nil), PChar(Format('File Can not be loaded (Error Code %d)', [GetLastError()])), 'error', MB_ICONERROR);
      Exit;
   end;

   hMap:= CreateFileMapping(hFile, NIL, PAGE_READONLY, 0, 0, NIL);
   pBase:= MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
   CloseHandle(hMap);
   hMap:= 0;
   CurAddr:= Cardinal(pBase);

   //read private header
   ReadRamFile(yy, 2);
   ReadRamFile(m, 1);
   ReadRamFile(d, 1);
   ReadRamFile(w, 1);
   ReadRamFile(h, 1);
   ReadRamFile(TotalRec, 4);

   SetLength(LetterCount, 128);
   ZeroMemory(LetterCount, 128*sizeof(LongWord));

   ReadRamFile(LetterCount[0], 4*128);
   ReadRamFile(imgType, 1);
   if(imgType <> itGray) then
      imgType := itBinary;
   ReadRamFile(Comments[0], 256);
   if( (w > 0) AND (h > 0)) then
      normal := true
   else
      normal := False;

   CurAddr:= Cardinal(pBase) + 1024;//bypass 1024 bytes header
   SetLength(Data, TotalRec);
   SetLength(labels, TotalRec);
   for i := 0 to TotalRec-1 do
   begin
	   ReadRamFile(StartByte, 1);//must be 0xff
	   ReadRamFile(labels[i], 1);
	   if (not normal) then
      begin
		   ReadRamFile(W, 1);
         ReadRamFile(H, 1);
      end;
	   ReadRamFile(ByteCount, 2);
	   SetLength(Data[i], H, W);

      if(imgType = itBinary) then
      begin
         for y:= 0 to H-1 do
         begin
            bWhite:= true;
            counter:= 0;
            while (counter < W) do
            begin
             ReadRamFile(WBcount, 1);
               x:= 0;
               while(x < WBcount) do
             begin
                  if(bWhite) then
                     Data[i][y, x+counter]:= 0//Background
                  else
                     Data[i][y, x+counter]:= 1;//ForeGround
                  x:= x+1;
             end;
             bWhite:= not bWhite;//black white black white ...
             counter:= counter + WBcount;
            end
         end;
      end
      else//GrayScale mode
      begin
         for y:= 0 to H-1 do
            for x:= 0 to W-1 do
               ReadRamFile(Data[i][y,x], 1);
      end;
   end;//i

   UnmapViewOfFile(pBase);
   CloseHandle(hFile); 
end;

procedure ReadRamFile(var Buffer; ByteCount:integer);
begin
   CopyMemory(@Buffer, Pointer(CurAddr), ByteCount);
   CurAddr:= CurAddr+ByteCount;
end;
end.
