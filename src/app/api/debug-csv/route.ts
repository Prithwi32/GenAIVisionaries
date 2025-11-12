import { NextRequest, NextResponse } from 'next/server'

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData()
    const file = formData.get('file') as File
    
    if (!file) {
      return NextResponse.json({ error: 'No file uploaded' }, { status: 400 })
    }
    
    const text = await file.text()
    const lines = text.split(/\r?\n/).filter(line => line.trim())
    
    // Simple CSV parsing
    const parseCSVLine = (line: string): string[] => {
      return line.split(',').map(cell => cell.trim().replace(/^"|"$/g, ''))
    }
    
    const header = parseCSVLine(lines[0])
    const sampleRows = lines.slice(1, 6).map(line => parseCSVLine(line)) // First 5 data rows
    
    return NextResponse.json({
      success: true,
      fileName: file.name,
      fileSize: file.size,
      totalLines: lines.length,
      header: header,
      sampleRows: sampleRows,
      firstLineRaw: lines[0],
      secondLineRaw: lines[1] || 'No data line'
    })
    
  } catch (error: any) {
    return NextResponse.json({
      success: false,
      error: error.message,
      stack: error.stack
    }, { status: 500 })
  }
}