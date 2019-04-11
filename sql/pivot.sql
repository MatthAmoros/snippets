USE [erpFrusys]
GO
/****** Object:  StoredProcedure [dbo].[SPC_CALIDAD_PIVOT]    Script Date: 11/04/2019 17:09:17 ******/
SET ANSI_NULLS OFF
GO
SET QUOTED_IDENTIFIER OFF
GO
ALTER PROCEDURE [dbo].[SPC_CALIDAD_PIVOT]
AS


DECLARE @cols NVARCHAR (MAX)

SELECT @cols = COALESCE (@cols + ',[' + ParameterName + ']', 
               '[' + ParameterName + ']')
               FROM    (SELECT DISTINCT ParameterName,0 O 
               FROM AT_COM_VOC_CALIDAD_PARAM
			   WHERE ParameterType <> 'ExportDurofel'
              ) PV  
               ORDER BY O


DECLARE @query NVARCHAR(MAX)
SET @query = '           
              SELECT * FROM 
             (
				SELECT [Codigo Productor]
					  ,[Especie]
					  ,[Codigo Variedad]
					  ,[Lote]
					  ,[FECHA]
					  ,[Grupo Productor]
					  ,[Nombre productor]
					  ,[Descripción]
					  ,[Nombre Variedad]
					  ,[VALOR]
					  ,[VALOR_NUERIC]
					  ,[TOTAL_KILOS]
					  ,[Hora recepción]
					  ,[KILOS]
					  ,[CANTIDAD]
					  ,[COD_FRI]
					  ,[HORA_REC]
				  FROM [erpFrusys].[dbo].[viewControlCalidad_AT_COM]
             ) x
             PIVOT 
             (
                 MIN(VALOR_NUERIC)
                 FOR [Descripción] IN (' + @cols + ')
            ) p    

            '     
EXEC SP_EXECUTESQL @query
