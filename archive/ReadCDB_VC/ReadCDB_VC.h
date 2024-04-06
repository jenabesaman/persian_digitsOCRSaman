// ReadCDB_VC.h : main header file for the PROJECT_NAME application
//

#pragma once

#ifndef __AFXWIN_H__
	#error include 'stdafx.h' before including this file for PCH
#endif

#include "resource.h"		// main symbols


// CReadCDB_VCApp:
// See ReadCDB_VC.cpp for the implementation of this class
//

class CReadCDB_VCApp : public CWinApp
{
public:
	CReadCDB_VCApp();

// Overrides
	public:
	virtual BOOL InitInstance();

// Implementation

	DECLARE_MESSAGE_MAP()
};

extern CReadCDB_VCApp theApp;